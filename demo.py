"""
CUDA-Lab Demo - Alle User Stories in 1 Bestand
===============================================
Student: Hilario Pedro Oliveira Sanca
Module: Code Exploration Lab (CUDA + GPU Programming)

Run: python3 demo.py
"""

import numpy as np
import math
import json

# Probeer Numba te importeren voor GPU, anders CPU fallback
try:
    from numba import vectorize
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️  Numba niet beschikbaar - draait alleen op CPU")


# ============================================================
# CUDA KERNEL - Dit is de GPU versnelling!
# ============================================================
if NUMBA_AVAILABLE:
    @vectorize(['float32(float32)'], target='cuda')
    def fast_exp(x):
        """GPU-versnelde exponential functie via Numba CUDA"""
        return math.exp(x)
else:
    def fast_exp(x):
        """CPU fallback"""
        return np.exp(x)


# ============================================================
# USER STORY 1: Dataset inladen
# ============================================================
def load_data():
    """Laad sample tekst data"""
    # Simpele Nederlandse voorbeeldtekst
    text = """een goede student leert elke dag. een student die leert wordt beter.
de docent helpt de student. de student helpt een andere student.
leren is belangrijk. oefening maakt de meester."""

    # Maak het langer voor betere training
    text = (text + " ") * 100
    return text


# ============================================================
# USER STORY 2: Tokeniseren
# ============================================================
def tokenize(text):
    """Maak vocabulaire en converteer tekst naar nummers"""
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int32)
    return data, stoi, itos


# ============================================================
# USER STORY 3-6: Model Training (CPU en GPU)
# ============================================================
def softmax(logits, use_gpu=False):
    """Softmax met optionele GPU versnelling"""
    if use_gpu and NUMBA_AVAILABLE:
        # GPU versnelde exp
        exp_vals = fast_exp(logits.astype(np.float32))
    else:
        # CPU versie
        exp_vals = np.exp(logits)

    return exp_vals / exp_vals.sum(axis=1, keepdims=True)


def train(data, vocab_size, epochs=50, learning_rate=0.1, use_gpu=False):
    """Train een simpel bigram model"""
    # Initialiseer gewichten
    W = np.random.randn(vocab_size, vocab_size).astype(np.float32) * 0.01

    # Training
    losses = []
    for epoch in range(epochs):
        # Maak training pairs (huidige -> volgende karakter)
        xs = data[:-1]
        ys = data[1:]

        # Forward pass
        logits = W[xs]
        probs = softmax(logits, use_gpu=use_gpu)

        # Loss (cross entropy)
        loss = -np.log(probs[np.arange(len(ys)), ys] + 1e-8).mean()
        losses.append(float(loss))

        # Backward pass (simpele gradient)
        dlogits = probs
        dlogits[np.arange(len(ys)), ys] -= 1
        dlogits /= len(ys)

        # Update weights
        for i in range(len(xs)):
            W[xs[i]] -= learning_rate * dlogits[i]

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: loss = {loss:.3f}")

    return W, losses


# ============================================================
# USER STORY 7-8: Benchmark CPU vs GPU
# ============================================================
def benchmark():
    """Meet snelheid CPU vs GPU voor exp() operatie"""
    import time

    n = 5_000_000
    x = np.random.randn(n).astype(np.float32)

    # CPU benchmark
    start = time.time()
    _ = np.exp(x)
    cpu_time = time.time() - start

    # GPU benchmark
    if NUMBA_AVAILABLE:
        start = time.time()
        _ = fast_exp(x)
        gpu_time = time.time() - start
        speedup = cpu_time / gpu_time
    else:
        gpu_time = cpu_time
        speedup = 1.0

    return cpu_time, gpu_time, speedup


# ============================================================
# USER STORY 9-10: Model opslaan en laden
# ============================================================
def save_model(W, stoi, itos):
    """Sla model op"""
    np.save('model.npy', W)
    with open('vocab.json', 'w') as f:
        json.dump({'stoi': stoi, 'itos': itos}, f)


def load_model():
    """Laad model"""
    W = np.load('model.npy')
    with open('vocab.json', 'r') as f:
        vocab = json.load(f)
    # Converteer itos keys terug naar integers
    itos = {int(k): v for k, v in vocab['itos'].items()}
    return W, vocab['stoi'], itos


# ============================================================
# USER STORY 11: Tekst genereren
# ============================================================
def generate(W, stoi, itos, start='e', length=100):
    """Genereer tekst met getraind model"""
    if start not in stoi:
        start = list(stoi.keys())[0]

    current = stoi[start]
    result = [start]

    for _ in range(length - 1):
        logits = W[current]
        probs = np.exp(logits) / np.exp(logits).sum()
        current = np.random.choice(len(probs), p=probs)
        result.append(itos[current])

    return ''.join(result)


# ============================================================
# USER STORY 12: Loss-grafiek bekijken
# ============================================================
def plot_losses(losses, bestand='loss.png'):
    """Toon het lossverloop als grafiek in de terminal en sla een PNG op"""
    # Grafiek in de terminal - werkt altijd, geen extra bibliotheek nodig
    hoogte = 8
    laagste, hoogste = min(losses), max(losses)
    bereik = (hoogste - laagste) or 1.0
    kolommen = losses[::max(1, len(losses) // 50)]

    for rij in range(hoogte, 0, -1):
        grens = laagste + bereik * rij / hoogte
        lijn = ''.join('*' if l >= grens else ' ' for l in kolommen)
        print(f"   {grens:6.3f} |{lijn}")
    print(f"          +{'-' * len(kolommen)}")
    print(f"           epoch 0 → {len(losses) - 1}")

    # PNG erbij, als matplotlib beschikbaar is
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 3))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (cross entropy)')
        plt.title('CUDA-Lab trainingsverloop')
        plt.tight_layout()
        plt.savefig(bestand, dpi=120)
        plt.close()
        return bestand
    except ImportError:
        return None


# ============================================================
# USER STORY 13: Duidelijke foutmeldingen bij ongeldige invoer
# ============================================================
def valideer_instellingen(epochs, learning_rate, lengte):
    """Controleer de hyperparameters en geef een begrijpelijke fout bij onzin"""
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError(
            f"epochs moet een positief geheel getal zijn, maar was {epochs!r}. "
            "Probeer bijvoorbeeld epochs=50."
        )
    if not 0 < learning_rate <= 1:
        raise ValueError(
            f"learning_rate moet tussen 0 en 1 liggen, maar was {learning_rate!r}. "
            "Een te grote waarde laat de training ontsporen; probeer 0.1."
        )
    if not isinstance(lengte, int) or lengte < 1:
        raise ValueError(
            f"lengte moet een positief geheel getal zijn, maar was {lengte!r}. "
            "Probeer bijvoorbeeld lengte=150."
        )
    return True


# ============================================================
# MAIN DEMO
# ============================================================
def main():
    print("\n" + "="*60)
    print("  CUDA-LAB DEMO - Code Exploration Lab")
    print("  Student: Hilario Pedro Oliveira Sanca")
    print("="*60 + "\n")

    # USER STORY 1-2: Data laden en tokenizen
    print("📁 USER STORY 1-2: Dataset laden + tokenizen")
    text = load_data()
    data, stoi, itos = tokenize(text)
    print(f"   ✅ Geladen: {len(text)} karakters")
    print(f"   ✅ Vocabulaire: {len(stoi)} unieke karakters")
    print()

    # USER STORY 3: Hyperparameters instellen
    print("⚙️  USER STORY 3: Hyperparameters instellen")
    epochs = 50
    learning_rate = 0.1
    lengte = 150
    valideer_instellingen(epochs, learning_rate, lengte)
    print(f"   ✅ epochs={epochs}, learning_rate={learning_rate}, lengte={lengte}")
    print()

    # USER STORY 4: CPU Training
    print("🔄 USER STORY 4: Training op CPU")
    W_cpu, losses_cpu = train(data, len(stoi), epochs=epochs,
                              learning_rate=learning_rate, use_gpu=False)
    print(f"   ✅ Loss: {losses_cpu[0]:.3f} → {losses_cpu[-1]:.3f}")
    print()

    # USER STORY 5-6: GPU Training
    print("🚀 USER STORY 5-6: Training met GPU (CUDA kernel)")
    if NUMBA_AVAILABLE:
        print("   ⚡ Gebruikt @vectorize(target='cuda') kernel")
    W_gpu, losses_gpu = train(data, len(stoi), epochs=epochs,
                              learning_rate=learning_rate, use_gpu=True)
    print(f"   ✅ Loss: {losses_gpu[0]:.3f} → {losses_gpu[-1]:.3f}")
    print()

    # USER STORY 7-8: Benchmark
    print("⏱️  USER STORY 7-8: Benchmark CPU vs GPU")
    cpu_time, gpu_time, speedup = benchmark()
    print(f"   CPU tijd:  {cpu_time*1000:6.2f} ms")
    print(f"   GPU tijd:  {gpu_time*1000:6.2f} ms")
    print(f"   ⚡ Speedup: {speedup:.1f}x")
    print()

    # USER STORY 9-10: Opslaan en laden
    print("💾 USER STORY 9-10: Model opslaan + laden")
    save_model(W_cpu, stoi, itos)
    print("   ✅ Model opgeslagen (model.npy, vocab.json)")
    W_loaded, stoi_loaded, itos_loaded = load_model()
    print(f"   ✅ Model geladen (shape: {W_loaded.shape})")
    print()

    # USER STORY 11: Genereren
    print("🎲 USER STORY 11: Tekst genereren")
    generated = generate(W_cpu, stoi, itos, start='e', length=lengte)
    print(f'   "{generated}"')
    print()

    # USER STORY 12: Loss-grafiek
    print("📈 USER STORY 12: Loss-grafiek bekijken")
    bestand = plot_losses(losses_cpu)
    if bestand:
        print(f"   ✅ Grafiek ook opgeslagen als {bestand}")
    else:
        print("   ℹ️  Matplotlib niet beschikbaar - alleen terminalgrafiek")
    print()

    # USER STORY 13: Duidelijke foutmeldingen
    print("🛡️  USER STORY 13: Duidelijke foutmeldingen bij ongeldige invoer")
    for slechte_waarde in [(-5, 0.1, 150), (50, 7.5, 150)]:
        try:
            valideer_instellingen(*slechte_waarde)
        except ValueError as fout:
            print(f"   ✅ Afgevangen: {fout}")
    print()

    # Samenvatting
    print("="*60)
    print("  ✅ ALLE MUST-HAVE EN SHOULD-HAVE STORIES GEDEMONSTREERD")
    print("="*60)
    print("""
    Must have (8/8)
    [1] ✅ Dataset inladen
    [2] ✅ Tokeniseren
    [3] ✅ Hyperparameters instellen
    [4] ✅ CPU training
    [5] ✅ GPU training (CUDA @vectorize kernel)
    [6] ✅ Loss volgen
    [7] ✅ CPU vs GPU benchmark
    [8] ✅ Speedup zien

    Should have (5/5)
    [9]  ✅ Model opslaan
    [10] ✅ Model laden
    [11] ✅ Tekst genereren
    [12] ✅ Loss-grafiek bekijken
    [13] ✅ Duidelijke foutmeldingen

    📈 Totaal: 13/13 user stories binnen scope (100%)
    ⚪ Could have (14-18) bewust niet opgepakt - MoSCoW
    """)


if __name__ == "__main__":
    main()