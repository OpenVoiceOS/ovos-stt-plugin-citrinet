# OVOS Citrinet STT


## Description

OpenVoiceOS STT plugin for [Nemo Citrinet](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#citrinet)

> **NOTE**: only onnx converted models can be used with this plugin

## Install

`pip install ovos-stt-plugin-citrinet`

## Configuration

```json
  "stt": {
    "module": "ovos-stt-plugin-citrinet",
    "ovos-stt-plugin-citrinet": {
        "lang": "ca"
    }
  }
```

## Credits

[NeonGeckoCom/streaming-stt-nemo](https://github.com/NeonGeckoCom/streaming-stt-nemo) - base citrinet onnx runtime implementation, provides [models](https://huggingface.co/collections/neongeckocom/neon-stt-663ca3c1a55b063463cb0167) for `'en', 'es', 'fr', 'de', 'it', 'uk', 'nl', 'pt'`

[Aina Project's Catalan STT model](https://huggingface.co/projecte-aina/stt-ca-citrinet-512) - was fine-tuned from a pre-trained Spanish stt-es-citrinet-512 model using the NeMo toolkit. It has around 36.5M parámeters and has been trained on Common Voice 11.0.

![img.png](img.png)
> This work is funded by the Ministerio para la Transformación Digital y de la Función Pública and Plan de Recuperación, Transformación y Resiliencia - Funded by EU – NextGenerationEU within the framework of the project ILENIA with reference 2022/TL22/00215337

![img_1.png](img_1.png)
> STT model was funded by the Generalitat de Catalunya within the framework of [Projecte AINA](https://politiquesdigitals.gencat.cat/ca/economia/catalonia-ai/aina).
