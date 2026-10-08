# inferencia.py — genera el archivo de envío desde el modelo guardado.
# Uso: python inferencia.py --modelo modelo_final --entrada submission.csv --salida submission.csv
import argparse, json, re, unicodedata
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def limpiar(t):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(t))).strip()

def construir(d, modo):
    texto = d["Comentario"].map(limpiar)
    if modo == "texto+metadatos":
        texto = "Sitio: " + d["Sitio"] + " | Lugar: " + d["Nombre del lugar"] + " | Valoración: " + d["Valoración"] + " | " + texto
    return texto.tolist()

def clasificar(modelo_dir, d, lote=32):
    meta = json.load(open(f"{modelo_dir}/entrenamiento.json"))["cfg"]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(modelo_dir)
    model = AutoModelForSequenceClassification.from_pretrained(modelo_dir).to(dev).eval()
    textos = construir(d, meta["entrada"])
    orden = sorted(range(len(textos)), key=lambda i: len(textos[i]))
    prob = np.zeros((len(textos), model.config.num_labels), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(orden), lote):
            idx = orden[i:i + lote]
            enc = tok([textos[j] for j in idx], truncation=True, max_length=meta["max_length"], padding=True, return_tensors="pt")
            logits = model(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev)).logits
            prob[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return prob

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True); ap.add_argument("--entrada", required=True); ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.entrada)
    prob = clasificar(a.modelo, d)
    pd.DataFrame({"ID": d["ID"], "Sentimiento": prob.argmax(1)}).to_csv(a.salida, index=False)
    print("filas:", len(d), "| predicciones:", np.bincount(prob.argmax(1), minlength=3).tolist())
