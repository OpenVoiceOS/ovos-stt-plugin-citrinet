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

<img src="img.png" width="128"/>

> This plugin was funded by the Ministerio para la Transformación Digital y de la Función Pública and Plan de Recuperación, Transformación y Resiliencia - Funded by EU – NextGenerationEU within the framework of the project ILENIA with reference 2022/TL22/00215337

<img src="img_1.png"  width="64"/>

> [projecte-aina/stt-ca-citrinet-512](https://huggingface.co/projecte-aina/stt-ca-citrinet-512) was funded by the Generalitat de Catalunya within the framework of [Projecte AINA](https://politiquesdigitals.gencat.cat/ca/economia/catalonia-ai/aina).

<img src="img_2.png"  width="64"/>

> [NeonGeckoCom/streaming-stt-nemo](https://github.com/NeonGeckoCom/streaming-stt-nemo) - base citrinet onnx runtime implementation, provides [models](https://huggingface.co/collections/neongeckocom/neon-stt-663ca3c1a55b063463cb0167) for `'en', 'es', 'fr', 'de', 'it', 'uk', 'nl', 'pt'`
