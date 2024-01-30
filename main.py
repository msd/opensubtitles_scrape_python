from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from pathlib import Path
import random
from time import sleep
import traceback
from xml.etree import ElementTree as ET
from typing import Iterable, TypeVar
from urllib.parse import urljoin

import requests
from requests import Session

from msd_common.download import download_file, downloading_mode

from languages import language  # type:ignore

T = TypeVar("T")


def drop(n: int, it: Iterable[T]) -> Iterable[T]:
    _dropped = list(zip(range(n), it))
    return it


def search_move_url(movie_name: str):
    return f"https://www.opensubtitles.org/en/search2/moviename-{movie_name}/sublanguageid-all/xml"


@dataclass
class XMLPage:
    url: str | None = None
    session: Session | None = None
    _xml: str | None = None

    @cached_property
    def xml(self):
        if self._xml is not None:
            return self._xml
        if self.url is not None:
            response = (self.session or requests).get(self.url)
            response.raise_for_status()
            return response.text
        raise RuntimeError("must either give a url or xml")

    @cached_property
    def tree(self):
        return ET.fromstring(self.xml)

    @staticmethod
    def from_url(url: str, session: Session):
        return XMLPage(url=url, session=session)

    @staticmethod
    def from_text(text: str):
        return XMLPage(_xml=text)


@dataclass
class InternetFile:
    url: str
    session: Session

    def download(self, out_path: Path):
        download_file(
            self.url,
            out_path,
            session=self.session,
            mode=downloading_mode.skip_existing,
        )


@dataclass
class Subtitle:
    """
    movie with subtitles of the same language as this subtitle
    Example="/en/search/sublanguageid-spa/idmovie-1155263"
    //subtitle/IDSubtitle/@MovieLink

    subtitle info page (html)
    Example="/en/subtitles/9533202/dungeons-dragons-honor-among-thieves-es"
    //subtitle/IDSubtitle/@Link

    Example="https://dl.opensubtitles.org/en/download/subad/9533202"
    //subtitle/IDSubtitle/@LinkDownload

    same as MovieLink
    Example="/en/search/sublanguageid-spa/idmovie-1155263"
    //subtitle/IDSubtitle/@LanguageLink

    movie imdb
    Example="/redirect/http://www.imdb.com/title/tt2906216/"
    //subtitle/IDSubtitle/@LinkImdb

    Example="9a1ae35a-7602-7968-9e2a-2d806fdfd154"
    //subtitle/IDSubtitle/@uuid

    subtitle id (found in Link and LinkDownload)
    format: integer
    Example=9533202
    //subtitle/IDSubtitle/text()
    """

    _page: XMLPage
    session: Session

    @property
    def _elem(self):
        return self._page.tree

    def _find(self, selector: str):
        return next(self._elem.iterfind(".//" + selector))

    @cached_property
    def _main(self):
        return next(self._sub_id_elem.iterfind(".."))

    def _mfind(self, selector: str):
        return next(self._main.iterfind(".//" + selector))

    @cached_property
    def _sub_id_elem(self):
        return self._find("Subtitle/IDSubtitle")

    @cached_property
    def sub_id(self):
        # example: 9641833
        # part of the subtitle page https://www.opensubtitles.org/en/subtitles/9641833/dungeons-dragons-honor-among-thieves-pb/xml
        id_: str = self._sub_id_elem.text  # type: ignore
        return id_

    @cached_property
    def fps(self):
        e = self._main.find("MovieFPS")
        assert e is not None
        frames: str = e.text  # type: ignore
        return frames

    @cached_property
    def zip_url(self):
        "download zip of subtitle and info with NO ads"

        # The main Download link from HTML page (gives status 301 but still offers file for download)
        # format:
        "https://www.opensubtitles.org/en/subtitleserve/sub/9641833"

        # The above link redirects to the following
        "https://dl.opensubtitles.org/en/download/sub/9641833"

        # _url = urljoin(HOME_PAGE, self._mfind("Download").attrib["LinkRedirect"])
        return f"https://dl.opensubtitles.org/en/download/sub/{self.sub_id}"

    @cached_property
    def zip_with_ads_url(self):
        # First link offered from XML page
        "https://dl.opensubtitles.org/en/download/subad/9641833"

        # _url = self._mfind("Download").attrib["LinkDownload"]
        return f"https://dl.opensubtitles.org/en/download/subad/{self.sub_id}"

    @cached_property
    def file_id(self):
        return self._mfind("SubtitleFile/File").attrib["ID"]

    @staticmethod
    def file_url_template(file_id: str):
        return f"https://dl.opensubtitles.org/en/download/file/{file_id}"

    @cached_property
    def file_url(self):
        # This link gives us 302 Redirect (but still offers the file for download)
        # https://www.opensubtitles.org/en/subtitleserve/file/1958209619
        # The redirect leads the following link
        # https://dl.opensubtitles.org/en/download/file/1958209619

        # _url = urljoin(HOME_PAGE, self._mfind("SubtitleFile/File/SubActualCD").attrib["LinkRedirect"])
        return Subtitle.file_url_template(self.file_id)

    @cached_property
    def file_with_ads_url(self):
        # Example: https://dl.opensubtitles.org/en/download/filead/1958209619
        # format:

        # _url = self._mfind("SubtitleFile/File/SubActualCD").attrib["LinkDownload"]
        return f"https://dl.opensubtitles.org/en/download/filead/{self.file_id}"

    @cached_property
    def movie_name(self):
        name: str = self._mfind("MovieReleaseName").text  # type: ignore
        return name

    @staticmethod
    def from_url(url: str, session: Session):
        # url = f"https://www.opensubtitles.org/en/subtitles/9641833/dungeons-dragons-honor-among-thieves-pb/xml"
        return Subtitle(_page=XMLPage.from_url(url, session=session), session=session)

    @staticmethod
    def file_from_file_id(file_id: str, session: Session):
        return InternetFile(url=Subtitle.file_url_template(file_id), session=session)


# //Download@LinkRedirect == download_url_template_actual
# Final link offered for download from xml page //Download@LinkDownloadBundle
"https://dl.opensubtitles.org/en/download/subb/9641833"
# //File@ID => file id appears in filead and subtitleserve


class movie_kind(Enum):
    series = "tv series"
    movie = "movie"
    episode = "episode"


HOME_PAGE = "https://www.opensubtitles.org/"


@dataclass
class MediaSearchItemInfo:
    _element: ET.Element

    def _find_one(self, selector: str):
        return next(self._element.iterfind(selector))

    @cached_property
    def thumb(self):
        u: str = self._find_one(".//MovieThumb").text  # type: ignore
        return urljoin(HOME_PAGE, u)

    @cached_property
    def name(self):
        return self._find_one(".//MovieName").text or "!"

    @cached_property
    def year(self):
        return self._find_one(".//MovieYear").text or "!"

    @cached_property
    def _id_elem(self):
        return self._find_one(".//MovieID")

    @cached_property
    def id(self):
        return self._id_elem.text

    # URL to XML page of Movie/Single Episode (aka subtitle index)
    @cached_property
    def url(self):
        return urljoin(HOME_PAGE, self._id_elem.attrib["Link"])

    @cached_property
    def imdb_link(self):
        return self._id_elem.attrib["LinkImdb"][len("/redirect/") :]

    @cached_property
    def movie_kind(self):
        return movie_kind(self._find_one("MovieKind").text)

    @cached_property
    def is_series(self):
        return self.movie_kind == movie_kind.series

    def __str__(self):
        return f"[{self.movie_kind.value.upper()}] [{self.id}] {self.name} ({self.year}) {self.imdb_link}"


@dataclass
class MediaSearchItem:
    info: MediaSearchItemInfo

    def media(self, session: Session):
        if self.info.is_series:
            raise RuntimeError("search result is a series")
        return Media.from_url(self.info.url, session=session)

    def series(self, session: Session):
        if not self.info.is_series:
            raise RuntimeError("search result is not a series")

        # TODO return ssearch url
        raise NotImplementedError()

    def __str__(self):
        return f"SEARCH ITEM: {self.info}"


@dataclass
class MediaSearchResults:
    url: str
    page: XMLPage
    session: Session

    @cached_property
    def results(self):
        # avoids ads that do not have any of these tags
        xpath = ".//results/subtitle/MovieID/.."

        return [
            MediaSearchItem(MediaSearchItemInfo(e))
            for e in self.page.tree.iterfind(xpath)
        ]

    @cached_property
    def prev_url(self):
        t = next(self.page.tree.iterfind(".//search/results_found/from"))
        if "Linker" in t.attrib:
            return urljoin(HOME_PAGE, t.attrib["Linker"])

    @cached_property
    def next_url(self):
        t = next(self.page.tree.iterfind(".//search/results_found/to"))
        if "Linker" in t.attrib:
            return urljoin(HOME_PAGE, t.attrib["Linker"])

    def has_next(self):
        return self.next_url is not None

    def has_prev(self):
        return self.prev_url is not None

    @cached_property
    def next_page(self):
        if self.next_url:
            return MediaSearchResults.from_url(self.next_url, self.session)

    @cached_property
    def prev_page(self):
        if self.prev_url:
            return MediaSearchResults.from_url(self.prev_url, self.session)

    @staticmethod
    def convert_name_for_api(name: str):
        return "+".join(name.strip().lower().split())

    @staticmethod
    def movie_search_url(name: str, language: language | None):
        converted = MediaSearchResults.convert_name_for_api(name)

        # This url searches for episode titles from series and movie titles
        # the html version also shows pictures for each search result (episode
        # preview or movie poster)
        lang = "all" if language is None else language.code3
        return f"https://www.opensubtitles.org/en/search2/sublanguageid-{lang}/moviename-{converted}/xml"

    @staticmethod
    def from_url(url: str, session: Session):
        return MediaSearchResults(
            url=url, page=XMLPage.from_url(url, session=session), session=session
        )

    @staticmethod
    def from_movie_search(
        name: str, session: Session, language: language | None = None
    ):
        return MediaSearchResults.from_url(
            MediaSearchResults.movie_search_url(name, language), session
        )


@dataclass
class Series:
    _page: XMLPage
    session: Session

    @staticmethod
    def url_template(idmovie: str):
        return "https://www.opensubtitles.org/en/ssearch/sublanguageid-por/idmovie-{idmovie}"

    @staticmethod
    def from_id(idmovie: str, session: Session):
        url = Series.url_template(idmovie)
        return Series.from_url(url, session)

    @staticmethod
    def from_url(url: str, session: Session):
        page = XMLPage(url=url, session=session)
        return Series(_page=page, session=session)


@dataclass
class MediaInfo:
    _page: XMLPage

    @cached_property
    def _name_elem(self):
        return self._find_one("search/Movie/MovieName")

    @cached_property
    def id(self):
        return self._name_elem.attrib["MovieID"]

    @cached_property
    def name(self):
        name_: str = self._name_elem.text  # type: ignore
        return name_

    @cached_property
    def imdb_id(self):
        return self._name_elem.attrib["MovieImdbID"]

    @cached_property
    def url(self):
        return urljoin(HOME_PAGE, self._name_elem.attrib["Link"])

    @cached_property
    def imdb_url(self):
        return self._name_elem.attrib["ImdbLink"]

    @cached_property
    def year(self):
        y: str = self._find_one("search/Movie/MovieYear").text  # type: ignore
        return int(y)

    @cached_property
    def plot(self):
        plot_: str = self._find_one("search/Movie/MoviePlot").text  # type: ignore
        return plot_

    @cached_property
    def thumbnail_url(self):
        t_url: str = self._find_one("search/Movie/MovieThumb").text  # type: ignore
        return t_url

    def _find(self, selector: str):
        return [e for e in self._page.tree.iterfind(".//" + selector)]

    def _find_one(self, selector: str):
        return next(self._page.tree.iterfind(".//" + selector))

    @cached_property
    def movie_kind(self):
        elem = self._find_one("MovieKind")
        return movie_kind(elem.text)

    @cached_property
    def language(self):
        lang = self._find_one("search/was_search").attrib["sublanguageid"]
        if lang != "all":
            return language.from_code3(lang)


@dataclass
class SubtitleIndexEntryInfo:
    _elem: ET.Element

    @cached_property
    def id(self):
        if (e := self._find_no_except("IDSubtitle")) is not None:
            return e.text

    @cached_property
    def author(self):
        if (e := self._find_no_except("UserNickName")) is not None:
            return e.text

    @cached_property
    def author_id(self):
        if (e := self._find_no_except("UserID")) is not None:
            return e.text

    @cached_property
    def fps(self):
        if (e := self._find_no_except("MovieFPS")) is not None:
            return e.text

    @cached_property
    def release_name(self):
        if (e := self._find_no_except("MovieReleaseName")) is not None:
            return e.text

    def _find(self, selector: str):
        return next(self._elem.iterfind(".//" + selector))

    def _find_no_except(self, selector: str):
        try:
            return self._find(selector)
        except StopIteration:
            pass

    @cached_property
    def file_id(self):
        id_: str = self._find("IDSubtitleFile").text  # type: ignore
        return id_

    @cached_property
    def language(self):
        lang: str = self._find("ISO639").text  # type: ignore
        return language.from_code2(lang)

    @cached_property
    def sub_url(self):
        return urljoin(HOME_PAGE, self._find("IDSubtitle").attrib["Link"])
        # alternatively f"https://www.opensubtitles.org/en/subtitles/{self.sub_id}"
        # also redirects to the same page

    def __str__(self):
        return f"[{self.id}] {self.release_name or 'subtitle'} ({self.language.code3})"

    @cached_property
    def file_url(self):
        return Subtitle.file_url_template(self.file_id)

    def file(self, session: Session):
        return Subtitle.file_from_file_id(self.file_id, session)


@dataclass
class SubtitleIndexEntry:
    info: SubtitleIndexEntryInfo
    session: Session

    def subtitle(self):
        return Subtitle.from_url(self.info.sub_url, session=self.session)

    def __str__(self):
        return str(self.info)


@dataclass
class Media:
    page: XMLPage
    session: Session

    _info: MediaInfo | None = None

    @staticmethod
    def from_url(url: str, session: Session):
        return Media(page=XMLPage.from_url(url, session=session), session=session)

    @staticmethod
    def from_movie_id(idmovie: str, session: Session, language: str | None):
        """idmovie: id of the media, can be a movie or a single episode of a series"""
        if language is not None:
            assert len(language) == 3
        lang = language or "all"
        url = f"https://www.opensubtitles.org/en/search/sublanguageid-{lang}/idmovie-{idmovie}/xml"
        # https://www.opensubtitles.org/en/search/moviename-dragons/sublanguageid-all/xml
        return Media.from_url(url, session)

    def _find(self, xpath_: str):
        return self.page.tree.iterfind(".//" + xpath_)

    def _find_one(self, xpath_: str):
        return next(self._find(xpath_))

    @cached_property
    def prev_url(self):
        t = self._find_one("search/results_found/from")
        if "Linker" in t.attrib:
            return urljoin(HOME_PAGE, t.attrib["Linker"])

    @cached_property
    def next_url(self):
        t = self._find_one("search/results_found/to")
        if "Linker" in t.attrib:
            return urljoin(HOME_PAGE, t.attrib["Linker"])

    def has_next(self):
        return self.next_url is not None

    def has_prev(self):
        return self.prev_url is not None

    @cached_property
    def next_page(self):
        if self.next_url:
            return Media(
                XMLPage.from_url(self.next_url, session=self.session),
                session=self.session,
                _info=self.info,
            )

    @cached_property
    def prev_page(self):
        if self.prev_url:
            return Media(
                XMLPage.from_url(self.prev_url, session=self.session),
                session=self.session,
                _info=self.info,
            )

    @cached_property
    def info(self):
        if self._info is not None:
            return self._info
        return MediaInfo(_page=self.page)

    def with_language(self, language: str):
        return Media.from_movie_id(
            self.info.id, language=language, session=self.session
        )

    def __str__(self):
        return f"{self.info.movie_kind.value.upper()}: [{self.info.id}] {self.info.name} ({self.info.year})"

    @cached_property
    def sub_index(self):
        return [
            SubtitleIndexEntry(SubtitleIndexEntryInfo(_elem=e), session=self.session)
            for e in self._find("subtitle/IDSubtitle/..")
        ]


class CharacterReplacePolicy(Enum):
    delete = auto()


BAD_WIN_CHARS = set(r"\/:*?\"<>|")


def is_good_path_char(c: str) -> bool:
    return c not in BAD_WIN_CHARS


def win_path_name(
    s: str,
    character_replace_policy: CharacterReplacePolicy = CharacterReplacePolicy.delete,
) -> str:
    CRP = CharacterReplacePolicy
    if character_replace_policy is CRP.delete:
        return "".join(filter(is_good_path_char, s))
    raise NotImplementedError("Character replace policy win_path_name")

def main():
    search_term = input("Please enter title: ")
    # search_term = "dungeons and dragons (2023)"
    session = requests.session()

    # fake mozilla headers
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.5",
        "Dnt": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:106.0) Gecko/20100101 Firefox/106.0",
    }
    session.headers.update(headers)

    # media = Media.from_movie_id("1155263", language=None, session=session)

    # e = media.page.tree

    search = MediaSearchResults.from_movie_search(
        search_term, session, language.english
    )
    print(*search.results, sep="\n")
    media: Media = search.results[0].media(session)
    print(*media.sub_index, sep="\n")

    WORKER_COUNT = 3
    FACTOR_REFILL = 3
    REFILL_MARK = WORKER_COUNT * FACTOR_REFILL

    pool = ThreadPoolExecutor(WORKER_COUNT)

    @dataclass
    class State:
        current_page: Media
        finished_queueing: bool
        futures: set[Future[None]]

    sub_folder = Path("subs")
    sub_folder.mkdir(exist_ok=True)

    lang_count = defaultdict[language, int](lambda: 0)

    def dl(sub: SubtitleIndexEntry):
        lang_count[sub.info.language] += 1
        # i = lang_count[sub.info.language]
        # file_name = f"{sub.info.language.code3}.{sub.info.language.name} {i} ({sub.info.id}).srt"
        file_name = f"{win_path_name(sub.info.release_name or '')} ({sub.info.id}).{sub.info.language.code3}.srt"
        sub.info.file(session).download(sub_folder / file_name)
        print("DOWNLOAD:", sub.info)
        delay = clamp(3, random.expovariate(1 / 6), 10)
        sleep(delay)

    fp = State(
        current_page=media,
        finished_queueing=False,
        futures={pool.submit(dl, sub) for sub in media.sub_index},
    )

    # add all subtitles to queue for downloading
    while not fp.finished_queueing:
        # remove done downloads
        for f in [f for f in fp.futures if f.done()]:
            if ex := f.exception(timeout=0.001):
                print(*traceback.format_exception(ex), sep="\n")
            fp.futures.remove(f)

        if len(fp.futures) < REFILL_MARK:
            if next_page := fp.current_page.next_page:
                fp.current_page = next_page
                fp.futures.update(
                    pool.submit(dl, sub) for sub in fp.current_page.sub_index
                )
            else:
                fp.finished_queueing = True
        sleep(0.2)

    # wait untill all done
    while fp.futures:
        # remove done downloads
        for f in [f for f in fp.futures if f.done()]:
            if ex := f.exception(timeout=0.001):
                print(*traceback.format_exception(ex), sep="\n")
            fp.futures.remove(f)
        sleep(0.5)


def clamp(lo: float, x: float, hi: float):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


if __name__ == "__main__":
    main()
