from enum import Enum
import enum


class language(Enum):
    abkhazian = "abk", "ab", "Abkhazian"
    afrikaans = "afr", "af", "Afrikaans"
    albanian = "alb", "sq", "Albanian"
    arabic = "ara", "ar", "Arabic"
    aragonese = "arg", "an", "Aragonese"
    armenian = "arm", "hy", "Armenian"
    assamese = "asm", "as", "Assamese"
    asturian = "ast", "at", "Asturian"
    azerbaijani = "aze", "az", "Azerbaijani"
    basque = "baq", "eu", "Basque"
    belarusian = "bel", "be", "Belarusian"
    bengali = "ben", "bn", "Bengali"
    bosnian = "bos", "bs", "Bosnian"
    breton = "bre", "br", "Breton"
    bulgarian = "bul", "bg", "Bulgarian"
    burmese = "bur", "my", "Burmese"
    catalan = "cat", "ca", "Catalan"
    chinese_simplified = "chi", "zh", "Chinese (simplified)"
    chinese_traditional = "zht", "zt", "Chinese (traditional)"
    chinese_bilingual = "zhe", "ze", "Chinese bilingual"
    croatian = "hrv", "hr", "Croatian"
    czech = "cze", "cs", "Czech"
    danish = "dan", "da", "Danish"
    dari = "prs", "pr", "Dari"
    dutch = "dut", "nl", "Dutch"
    english = "eng", "en", "English"
    esperanto = "epo", "eo", "Esperanto"
    estonian = "est", "et", "Estonian"
    extremaduran = "ext", "ex", "Extremaduran"
    finnish = "fin", "fi", "Finnish"
    french = "fre", "fr", "French"
    gaelic = "gla", "gd", "Gaelic"
    galician = "glg", "gl", "Galician"
    georgian = "geo", "ka", "Georgian"
    german = "ger", "de", "German"
    greek = "ell", "el", "Greek"
    hebrew = "heb", "he", "Hebrew"
    hindi = "hin", "hi", "Hindi"
    hungarian = "hun", "hu", "Hungarian"
    icelandic = "ice", "is", "Icelandic"
    igbo = "ibo", "ig", "Igbo"
    indonesian = "ind", "id", "Indonesian"
    interlingua = "ina", "ia", "Interlingua"
    irish = "gle", "ga", "Irish"
    italian = "ita", "it", "Italian"
    japanese = "jpn", "ja", "Japanese"
    kannada = "kan", "kn", "Kannada"
    kazakh = "kaz", "kk", "Kazakh"
    khmer = "khm", "km", "Khmer"
    korean = "kor", "ko", "Korean"
    kurdish = "kur", "ku", "Kurdish"
    latvian = "lav", "lv", "Latvian"
    lithuanian = "lit", "lt", "Lithuanian"
    luxembourgish = "ltz", "lb", "Luxembourgish"
    macedonian = "mac", "mk", "Macedonian"
    malay = "may", "ms", "Malay"
    malayalam = "mal", "ml", "Malayalam"
    manipuri = "mni", "ma", "Manipuri"
    marathi = "mar", "mr", "Marathi"
    mongolian = "mon", "mn", "Mongolian"
    montenegrin = "mne", "me", "Montenegrin"
    navajo = "nav", "nv", "Navajo"
    nepali = "nep", "ne", "Nepali"
    northern_sami = "sme", "se", "Northern Sami"
    norwegian = "nor", "no", "Norwegian"
    occitan = "oci", "oc", "Occitan"
    odia = "ori", "or", "Odia"
    persian = "per", "fa", "Persian"
    polish = "pol", "pl", "Polish"
    portuguese = "por", "pt", "Portuguese"
    portuguese_br = "pob", "pb", "Portuguese (BR)"
    portuguese_mz = "pom", "pm", "Portuguese (MZ)"
    pushto = "pus", "ps", "Pushto"
    romanian = "rum", "ro", "Romanian"
    russian = "rus", "ru", "Russian"
    santali = "sat", "sx", "Santali"
    serbian = "scc", "sr", "Serbian"
    sindhi = "snd", "sd", "Sindhi"
    sinhalese = "sin", "si", "Sinhalese"
    slovak = "slo", "sk", "Slovak"
    slovenian = "slv", "sl", "Slovenian"
    somali = "som", "so", "Somali"
    spanish = "spa", "es", "Spanish"
    spanish_eu = "spn", "sp", "Spanish (EU)"
    spanish_la = "spl", "ea", "Spanish (LA)"
    swahili = "swa", "sw", "Swahili"
    swedish = "swe", "sv", "Swedish"
    syriac = "syr", "sy", "Syriac"
    tagalog = "tgl", "tl", "Tagalog"
    tamil = "tam", "ta", "Tamil"
    tatar = "tat", "tt", "Tatar"
    telugu = "tel", "te", "Telugu"
    thai = "tha", "th", "Thai"
    toki_Pona = "tok", "tp", "Toki Pona"
    turkish = "tur", "tr", "Turkish"
    turkmen = "tuk", "tk", "Turkmen"
    ukrainian = "ukr", "uk", "Ukrainian"
    urdu = "urd", "ur", "Urdu"
    vietnamese = "vie", "vi", "Vietnamese"
    welsh = "wel", "cy", "Welsh"

    @enum.property
    def name(self) -> str:
        return self.value[2]

    @enum.property
    def code2(self) -> str:
        return self.value[1]

    @enum.property
    def code3(self) -> str:
        return self.value[0]

    @staticmethod
    def from_code2(code2: str):
        return lang_index_2[code2]

    @staticmethod
    def from_code3(code3: str):
        return lang_index_3[code3]

    @staticmethod
    def from_name(name: str):
        return lang_index_name[name]

    def __str__(self):
        return f"Language<{self.code3}, {self.name}>"


lang_index_3 = {e.code3: e for e in list(language)}
lang_index_2 = {e.code2: e for e in list(language)}
lang_index_name = {e.name: e for e in list(language)}
