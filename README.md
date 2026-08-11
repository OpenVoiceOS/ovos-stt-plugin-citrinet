> **⚠️ This plugin is archived.** Use
> [ovos-stt-plugin-onnx-asr](https://github.com/TigreGotico/ovos-stt-plugin-onnx-asr)
> instead — it runs the same Citrinet models (and more NeMo architectures) through
> [onnx-asr](https://github.com/istupakov/onnx-asr) without pytorch or NeMo.
> Ready-to-use ONNX Citrinet models for 13+ languages live in the
> [OpenVoiceOS STT/ASR collection](https://huggingface.co/collections/OpenVoiceOS/stt-asr-onnx-699321e8732462509c642fbe).

# OVOS Citrinet STT

## Description

This plugin gives OpenVoiceOS speech-to-text with [Nemo Citrinet](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#citrinet) models.

> **NOTE**: only onnx converted models work with this plugin

## Install

By default this plugin installs the full pytorch package. To avoid the extra dependencies, install the CPU-only version of pytorch **before** you install the plugin.

`pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu`

If you skip this step, pip installs the full pytorch package together with the plugin.

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

### Models

The table shows the self-reported WER score from each model page. A lower score means better accuracy.

| Model                                         | CV10  | CV11  | CV12   | LibriSpeech (clean) |  
|-----------------------------------------------|-------|-------|--------|---------------------| 
| `projecte-aina/stt-ca-citrinet-512`           |       | 6.684 |        |                     |  
| `neongeckocom/stt_ca_citrinet_512_gamma_0_25` |       |       | 8.065  |                     |  
| `neongeckocom/stt_es_citrinet_512_gamma_0_25` |       |       | 9.549  |                     |   
| `neongeckocom/stt_pt_citrinet_512_gamma_0_25` |       |       | 6.033  |                     |  
| `neongeckocom/stt_fr_citrinet_512_gamma_0_25` |       |       | 14.900 |                     |  
| `neongeckocom/stt_de_citrinet_512_gamma_0_25` |       |       | 11.100 |                     |  
| `neongeckocom/stt_it_citrinet_512_gamma_0_25` |       |       | 9.232  |                     |  
| `neongeckocom/stt_uk_citrinet_512_gamma_0_25` | 8.609 |       |        |                     |  
| `neongeckocom/stt_nl_citrinet_512_gamma_0_25` |       |       | 6.204  |                     |  
| `neongeckocom/stt_en_citrinet_512_gamma_0_25` |       |       |        | 3.400               |  

## Related projects

[OpenVoiceOS/ovos-stt-plugin-onnx-asr](https://github.com/OpenVoiceOS/ovos-stt-plugin-onnx-asr) is another OVOS STT plugin that runs onnx models.

## Credits

This plugin was developed by [TigreGotico](https://tigregotico.pt) for OpenVoiceOS under the [ILENIA](https://proyectoilenia.es) project.

<img src="img.png" width="128"/>

> This plugin was funded by the Ministerio para la Transformación Digital y de la Función Pública and Plan de
> Recuperación, Transformación y Resiliencia - Funded by EU – NextGenerationEU within the framework of the project
> [ILENIA](https://proyectoilenia.es)
> with reference 2022/TL22/00215337

<img src="img_1.png"  width="64"/>

> [projecte-aina/stt-ca-citrinet-512](https://huggingface.co/projecte-aina/stt-ca-citrinet-512) was funded by the
> Generalitat de Catalunya within the framework
> of [Projecte AINA](https://politiquesdigitals.gencat.cat/ca/economia/catalonia-ai/aina).

<img src="img_2.png"  width="64"/>

> [NeonGeckoCom/streaming-stt-nemo](https://github.com/NeonGeckoCom/streaming-stt-nemo) - base citrinet onnx runtime
> implementation, provides [models](https://huggingface.co/collections/neongeckocom/neon-stt-663ca3c1a55b063463cb0167)
> for `'en', 'es', 'fr', 'de', 'it', 'uk', 'nl', 'pt'`
