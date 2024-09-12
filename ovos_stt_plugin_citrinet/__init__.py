from typing import Optional, Dict

import numpy as np
from ovos_plugin_manager.templates.stt import STT
from ovos_utils.log import LOG
from speech_recognition import AudioData
from streaming_stt_nemo import Model, available_languages


class CitrinetSTT(STT):

    def __init__(self, config: dict = None):
        super().__init__(config)
        # replace default Neon model with project aina model
        Model.langs["ca"]["model"] = "projecte-aina/stt-ca-citrinet-512"
        self.lang = self.config.get('lang') or "ca"
        self.models: Dict[str, Model] = {}
        lang = self.lang.split("-")[0]
        if lang not in available_languages:
            raise ValueError(f"unsupported language, must be one of {available_languages}")
        LOG.info(f"preloading model: {Model.langs[lang]}")
        self.load_model(lang)

    def load_model(self, lang: str):
        if lang not in self.models:
            self.models[lang] = Model(lang=lang)
        return self.models[lang]

    @property
    def available_languages(self) -> set:
        return set(available_languages)

    def execute(self, audio: AudioData, language: Optional[str] = None):
        '''
        Executes speach recognition

        Parameters:
                    audio : input audio file path
        Returns:
                    text (str): recognized text
        '''
        lang = language.split("-")[0]
        if lang not in available_languages:
            raise ValueError(f"unsupported language, must be one of {available_languages}")
        model = self.load_model(lang)

        audio_buffer = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
        transcriptions = model.stt(audio_buffer, audio.sample_rate)

        if not transcriptions:
            LOG.debug("Transcription is empty")
            return None
        return transcriptions[0]


if __name__ == "__main__":

    b = CitrinetSTT({"lang": "ca"})
    from speech_recognition import Recognizer, AudioFile

    jfk = "/home/miro/PycharmProjects/ovos-stt-plugin-vosk/example.wav"
    with AudioFile(jfk) as source:
        audio = Recognizer().record(source)

    a = b.execute(audio, language="ca")
    print(a)
    # bon dia em dic abram orriols i garcia vaig néixer el vint de desembre del mil noucents norantasis a berga i sóc periodista
    # bon dia em dic abramriols i garcia vaig néixer el vint de desembre del mil nou-cents noranta-sis a berga i sóc periodista

