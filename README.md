# CUDA-Lab

**Student:** Hilario Pedro Oliveira Sanca  
**Module:** Code Exploration Lab (40401-KECOEX-01)  
**Onderwerp:** CUDA GPU-versnelling met Numba

---

## 🎯 Wat is dit?

Een **GPU-versnelde language model trainer** die CUDA fundamentals demonstreert.

**Type:** Character-level bigram neuraal netwerk  
**Doel:** CUDA leren + fundament voor LLM-training

---

## 🚀 Snel starten

```bash
python3 demo.py
```

Dat is het! Eén bestand, alle user stories.

---

## ✅ Wat het doet (User Stories)

**8 Must-have:**
1. Dataset inladen ✅
2. Tokeniseren ✅
3. Model configureren ✅
4. CPU training ✅
5. **GPU training** (CUDA @vectorize kernel) ⭐
6. Loss tracking ✅
7. CPU vs GPU benchmark ✅
8. Speedup meten ✅

**3 Should-have:**
9. Model opslaan ✅
10. Model laden ✅
11. Tekst genereren ✅

**Score:** 11/13 user stories (85%)

---

## 🔑 CUDA Feature

**De GPU kernel (regel ~30 in demo.py):**
```python
@vectorize(['float32(float32)'], target='cuda')
def fast_exp(x):
    return math.exp(x)
```

Dit versnelt de `exp()` operatie in de softmax functie.

---

## 📊 Wat je ziet

```
🔄 Training op CPU
  Loss: 3.178 → 3.077

🚀 Training met GPU  
  Loss: 3.178 → 3.077

⏱️  Benchmark
  CPU:  10 ms
  GPU:  10 ms (of sneller met echte NVIDIA GPU)
  
🎲 Gegenereerde tekst
  "een student leert..."
```

---

## 📦 Dependencies

```bash
pip install numpy numba
```

Werkt ook zonder Numba (CPU-only fallback).

---

## 🎓 Voor het assessment

1. Run `python3 demo.py`
2. Laat zien: alle 8 Must-have user stories
3. Toon de CUDA kernel code (regel ~30)
4. Klaar!

---

**Certificaat:** NVIDIA DLI - "Fundamentals of Accelerated Computing with CUDA Python"
