import os
from typing import Optional, Dict

from ovos_config import Configuration
from ovos_plugin_manager.templates.stt import STT
from ovos_plugin_manager.utils.audio import AudioData, AudioFile
from ovos_utils import classproperty
from ovos_utils.log import LOG

from ovos_stt_plugin_citrinet.engine import Model


class CitrinetSTT(STT):

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.models: Dict[str, Model] = {}  # lang: Model

        self.lang = self.config.get('lang') or Configuration().get("lang", "en")
        self.model_id = self.config.get('model')

        lang = self.lang.split("-")[0]
        if self.model_id and os.path.exists(self.model_id):
            LOG.info(f"loading local model: {self.model_id}")
            self.models[self.lang] = Model(lang=lang, model_folder=self.model_id)
        elif self.model_id:
            LOG.info(f"loading hf model: {self.model_id}")
            self.models[self.lang] = Model(lang=lang, hf_model=self.model_id)
        else:
            LOG.info(f"loading default lang model: {lang}")
            if lang not in self.available_languages:
                raise ValueError(f"unsupported language '{lang}', must be one of {self.available_languages}")
            self.models[lang] = Model(lang=lang)

    def load_model(self, lang: str):
        if lang not in self.models:
            self.models[lang] = Model(lang=lang)
        return self.models[lang]

    @classproperty
    def available_languages(cls) -> set:
        return set(Model.default_models.keys())

    def execute(self, audio: AudioData, language: Optional[str] = None):
        '''
        Executes speach recognition

        Parameters:
                    audio : input audio file path
        Returns:
                    text (str): recognized text
        '''
        language = language or self.lang
        lang = language.split("-")[0]
        if lang not in self.available_languages:
            raise ValueError(f"unsupported language '{lang}', must be one of {self.available_languages}")
        model = self.load_model(lang)

        audio_buffer = audio.get_np_float32(model.sample_rate)
        transcriptions = model.stt(audio_buffer)

        if not transcriptions:
            LOG.debug("Transcription is empty")
            return None
        return transcriptions[0]


if __name__ == "__main__":
    b = CitrinetSTT({"lang": "en"})

    jfk = "/home/miro/PycharmProjects/ovos-stt-plugin-fasterwhisper/jfk.wav"
    with AudioFile(jfk) as source:
        audio = source.read()

    a = b.execute(audio, language="en")
    print(a)
    # and so my fellow american ask not what your country can do for you ask what you can do for your country