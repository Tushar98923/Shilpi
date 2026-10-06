---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
language:
- en
tags:
- blender
- function-calling
- gguf
- llama.cpp
pipeline_tag: text-generation
---

# Shilpi operator model (Qwen3.5-2B, q8_0 GGUF)

The model behind Shilpi, an offline AI add-on for Blender: it turns a plain-English Blender command plus a compact
scene description into Blender operator calls. Runs locally with llama.cpp; the add-on downloads and starts it by
itself.

**Base model:** [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) by the Qwen team, licensed under Apache 2.0.
This fine-tuned model is also released under Apache 2.0.

## Results

Held-out test set: 196 commands not seen in training. Each answer is run in Blender 5.2.1 and checked against the
expected result.

| Model | Held-out accuracy |
|---|---|
| **This model** | **95.9%** |
| Qwen3.5-2B, not fine-tuned, same prompt | 4.6% |
| DeepSeek V4 Pro, same prompt | 12.8% |
| DeepSeek V4 Pro, same prompt plus argument names | 83.2% |

The local models use constrained decoding (a JSON schema of the supported operators); without it the untuned
Qwen3.5-2B scores 0% and this model scores the same 95.9%. Full setup: see the Shilpi repository's README.

## Format

System prompt:

```
You are the Blender operator agent. Turn the request into Blender operator calls. Reply with JSON only: {"calls": [{"op": "<bpy.ops name>", "args": {...}}]}. Use bai.select to select objects by name and bai.set to set location, rotation_euler (radians), scale, name, hidden or hide_render.
```

User message: `Scene: <compact scene JSON>` on one line, then `Request: <command>`. Thinking must be off
(`enable_thinking: false`). Example answer:

```json
{"calls": [{"op": "bai.select", "args": {"objects": ["Cube"]}}, {"op": "object.modifier_add", "args": {"type": "BEVEL"}}]}
```

Besides 245 real `bpy.ops` operators it uses helpers that the add-on implements: `bai.select`, `bai.set`,
`bai.loopcut`, `bai.keyframe`, `bai.frame`, `bai.collection`, `bai.ask` (asks which of two look-alike objects you
mean) and `bai.decline` (refuses requests it shouldn't act on).

## Training

LoRA fine-tuning (rank 32, 2 epochs) on 19,912 examples of synthetic instruction data: natural-language commands
paired with Blender operator calls. Every example was verified by executing it in Blender 5.2.1.

## Limitations

Single actions and short two-step commands; English only; trained on Blender 5.2. It sometimes refuses when the
user's word isn't the object's name ("donut" for an object named Torus), and occasionally forgets to select the
object first.
