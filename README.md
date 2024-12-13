# OVOS Citrinet STT

## Description

OpenVoiceOS STT plugin
for [Nemo Citrinet](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#citrinet)

> **NOTE**: only onnx converted models can be used with this plugin

## Install

by default this plugin will install the full pytorch, to avoid dragging all the dependencies it is recommended you install the cpu only version of pytorch **before** installing the plugin

`pip install torch==2.1.0+cpu  -f https://download.pytorch.org/whl/torch_stable.html`

If you skip the step above then the full pytorch will be installed together with the plugin

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

self reported WER score from model pages (lower is better)

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

## Credits

<img src="img.png" width="128"/>

> This plugin was funded by the Ministerio para la Transformación Digital y de la Función Pública and Plan de
> Recuperación, Transformación y Resiliencia - Funded by EU – NextGenerationEU within the framework of the project
> ILENIA
> with reference 2022/TL22/00215337

<img src="img_1.png"  width="64"/>

> [projecte-aina/stt-ca-citrinet-512](https://huggingface.co/projecte-aina/stt-ca-citrinet-512) was funded by the
> Generalitat de Catalunya within the framework
> of [Projecte AINA](https://politiquesdigitals.gencat.cat/ca/economia/catalonia-ai/aina).

<img src="img_2.png"  width="64"/>

> [NeonGeckoCom/streaming-stt-nemo](https://github.com/NeonGeckoCom/streaming-stt-nemo) - base citrinet onnx runtime
> implementation, provides [models](https://huggingface.co/collections/neongeckocom/neon-stt-663ca3c1a55b063463cb0167)
> for `'en', 'es', 'fr', 'de', 'it', 'uk', 'nl', 'pt'`
