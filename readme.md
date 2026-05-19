# Cheatsheet Transformers — PLN 

> Guía rápida para programar y entender los ejercicios de Transformers.  
> Orientada a examen: conceptos, código base y cuándo usar cada cosa.

# 1. Idea general de un Transformer

Un Transformer trabaja con texto convertido a números.

```text
Texto
↓
Tokenización
↓
input_ids + attention_mask
↓
Embeddings
↓
Transformer
↓
Salida
```

Según la tarea, se usa una arquitectura distinta:

| Arquitectura | Cuándo se usa | Ejemplo |
|---|---|---|
| **Encoder** | Entender/clasificar texto | Sentimiento, fake news, BERT |
| **Decoder** | Generar texto token a token | Predicción siguiente palabra, GPT |
| **Encoder + Decoder** | Transformar una secuencia en otra | Traducción, resumen |

---

# 2. Encoder, Decoder y Encoder-Decoder

## Encoder

Sirve para **entender una entrada completa**.

```text
Texto → Encoder → Vector/representación → Clasificación
```

Ejemplos:

- Clasificación de sentimiento.
- Fake news.
- Spam.
- Obtener embeddings contextualizados.

Modelos típicos:

```text
BERT, RoBERTa, DistilBERT
```

---

## Decoder

Sirve para **generar tokens**.

```text
Texto inicial → Decoder → siguiente token → siguiente token...
```

Ejemplos:

- Generación de texto.
- Autocompletado.
- Predicción del siguiente token.

Modelos típicos:

```text
GPT, LLaMA, Mistral
```

---

## Encoder + Decoder

Sirve para **convertir una secuencia en otra**.

```text
Frase origen → Encoder → representación
                              ↓
                          Decoder → frase destino
```

Ejemplos:

- Traducción.
- Resumen.
- Corrección gramatical.

Modelos típicos:

```text
T5, BART, MarianMT
```

---

# 3. Tokenización

El modelo no entiende texto directamente. Primero hay que convertirlo en tokens.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Transformers changed NLP."

tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)

print(tokens)
print(ids)
```

---

# 4. Tokens especiales

| Token | Significado | Función |
|---|---|---|
| `[CLS]` | Classification | Inicio de secuencia. Útil en clasificación |
| `[SEP]` | Separator | Final o separación de secuencias |
| `[PAD]` | Padding | Relleno para igualar longitudes |
| `[UNK]` | Unknown | Token desconocido |
| `<OOV>` | Out Of Vocabulary | Token fuera de vocabulario en Keras |

Ejemplo BERT:

```text
[CLS] transformers changed nl ##p . [SEP]
```

---

# 5. WordPiece

BERT usa WordPiece.

Si una palabra no está completa en el vocabulario, la divide en subtokens.

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

word = "internationalization"
tokens = tokenizer.tokenize(word)

print(tokens)
```

Ejemplo:

```text
internationalization → ['international', '##ization']
```

El prefijo `##` indica que el subtoken continúa la palabra anterior.

---

# 6. `input_ids` y `attention_mask`

Cuando tokenizamos con Hugging Face:

```python
encoding = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=12,
    return_tensors="pt"
)
```

obtenemos:

```python
encoding["input_ids"]
encoding["attention_mask"]
```

## `input_ids`

Son los identificadores numéricos de los tokens.

```text
[CLS] dogs are pets . [SEP] [PAD] [PAD]
↓
[101, 6077, 2024, 18551, 1012, 102, 0, 0]
```

## `attention_mask`

Indica qué posiciones debe mirar el modelo.

```text
1 → token real
0 → padding
```

Ejemplo:

```text
Tokens:         [CLS] dogs are pets . [SEP] [PAD] [PAD]
Attention mask:   1    1    1   1   1    1     0     0
```

---

# 7. Padding y truncation

## Padding

Rellena secuencias cortas.

```python
tokenizer(
    texts,
    padding=True
)
```

Rellena hasta la frase más larga del batch.

```python
tokenizer(
    texts,
    padding="max_length",
    max_length=12
)
```

Rellena siempre hasta longitud fija.

| Opción | Qué hace |
|---|---|
| `padding=True` | Rellena hasta el máximo del batch |
| `padding="max_length"` | Rellena hasta `max_length` |

---

## Truncation

Recorta secuencias largas.

```python
tokenizer(
    text,
    truncation=True,
    max_length=128
)
```

Si `max_length` es muy pequeño, se pierde información.

Ejemplo extremo:

```python
tokenizer(
    text,
    truncation=True,
    max_length=4
)
```

Resultado posible:

```text
[CLS] transformers changed [SEP]
```

---

# 8. Embeddings

Un embedding es una representación vectorial.

```text
palabra/frase → vector de números
```

Ejemplo:

```text
dog → [0.23, -0.11, 0.84, ...]
```

---

# 9. Embeddings estáticos vs contextualizados

| Tipo | Explicación | Ejemplos |
|---|---|---|
| **Estático** | Una palabra tiene siempre el mismo vector | Word2Vec, FastText, GloVe |
| **Contextualizado** | El vector cambia según la frase | BERT, ELMo, GPT |

Ejemplo con `bank`:

```text
I went to the bank to get money.
I sat near the river bank.
```

Word2Vec da casi el mismo vector para `bank`.

BERT genera vectores distintos según el contexto.

---

# 10. Embedding posicional

El Transformer por sí solo no sabe el orden de los tokens.

Necesita información de posición.

```text
Embedding final = embedding del token + embedding de posición
```

En Keras:

```python
import keras_nlp

token_embeddings = Embedding(vocab_size, embedding_dim)(inputs)

position_embeddings = keras_nlp.layers.PositionEmbedding(
    sequence_length=max_length
)(token_embeddings)

x = token_embeddings + position_embeddings
```

## Sin posición vs con posición

| Versión | Qué sabe |
|---|---|
| Sin embedding posicional | Qué tokens aparecen |
| Con embedding posicional | Qué tokens aparecen y dónde aparecen |

---

# 11. Transformer encoder básico en Keras

Para clasificación binaria:

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention
from tensorflow.keras.layers import LayerNormalization, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model

def create_transformer_classifier(vocab_size, max_length):
    inputs = Input(shape=(max_length,), dtype=tf.int32)

    x = Embedding(vocab_size, 32)(inputs)

    attention = MultiHeadAttention(
        num_heads=2,
        key_dim=32
    )(x, x, x)

    x = LayerNormalization()(attention + x)

    x = GlobalAveragePooling1D()(x)

    outputs = Dense(1, activation="sigmoid")(x)

    return Model(inputs, outputs)
```

Compilación:

```python
model = create_transformer_classifier(vocab_size, max_length)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

---

# 12. Qué son `vocab_size` y `max_length`

| Parámetro | Significado |
|---|---|
| `vocab_size` | Número máximo de tokens distintos |
| `max_length` | Longitud fija de cada secuencia |

Ejemplo:

```python
vocab_size = 10000
max_length = 100
embedding_dim = 32
```

Significa:

```text
Hay 10000 tokens posibles.
Cada texto tendrá 100 posiciones.
Cada token se representa con un vector de 32 dimensiones.
```

La salida de embedding tendrá forma:

```text
(batch_size, max_length, embedding_dim)
```

Ejemplo:

```text
(32, 100, 32)
```

---

# 13. MultiHeadAttention

La atención compara tokens entre sí.

```python
attention = MultiHeadAttention(
    num_heads=2,
    key_dim=32
)(x, x, x)
```

En un encoder se usa self-attention:

```text
query = x
key = x
value = x
```

Por eso aparece:

```python
(x, x, x)
```

---

# 14. Conexión residual y normalización

```python
x = LayerNormalization()(attention + x)
```

Esto hace:

```text
salida = normalizar(entrada original + atención)
```

Sirve para:

- Evitar perder la información original.
- Estabilizar el entrenamiento.
- Facilitar redes más profundas.

---

# 15. GlobalAveragePooling1D

En clasificación necesitamos convertir una secuencia en un único vector.

Antes:

```text
(max_length, embedding_dim)
```

Después:

```text
(embedding_dim,)
```

Código:

```python
x = GlobalAveragePooling1D()(x)
```

Luego:

```python
outputs = Dense(1, activation="sigmoid")(x)
```

---

# 16. Decoder Transformer básico

El decoder se usa para predecir tokens.

Diferencia clave:

```python
use_causal_mask=True
```

Esto impide mirar tokens futuros.

```python
def create_transformer_decoder(vocab_size, max_length, embedding_dim):
    inputs = Input(shape=(max_length,), dtype=tf.int32)

    x = Embedding(vocab_size, embedding_dim)(inputs)

    attention = MultiHeadAttention(
        num_heads=2,
        key_dim=embedding_dim
    )(
        x, x, x,
        use_causal_mask=True
    )

    x = LayerNormalization()(attention + x)

    outputs = Dense(vocab_size)(x)

    return Model(inputs, outputs)
```

Compilación:

```python
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
```

---

# 17. Preparar datos para decoder

Frase original:

```text
[w1, w2, w3, w4]
```

Entrada:

```text
X = [w1, w2, w3]
```

Salida esperada:

```text
Y = [w2, w3, w4]
```

Código:

```python
X = []
Y = []

for seq in sequences:
    X.append(seq[:-1])
    Y.append(seq[1:])
```

Padding:

```python
X = pad_sequences(X, maxlen=max_length)
Y = pad_sequences(Y, maxlen=max_length)
```

---

# 18. Predecir siguiente token

```python
def predict_next_token(model, tokenizer, sentence, max_length):
    seq = tokenizer.texts_to_sequences([sentence])[0]
    seq = pad_sequences([seq], maxlen=max_length)

    predictions = model.predict(seq)

    last_pred_logits = predictions[0, -1]

    next_token_id = np.argmax(last_pred_logits)

    return tokenizer.index_word.get(next_token_id, None)
```

---

# 19. Encoder-Decoder para traducción

Se usa en traducción:

```text
hello world → hola mundo
```

## Entrada encoder

Frase en inglés:

```python
encoder_inputs
```

## Entrada decoder

Frase en español sin último token:

```text
<bos> hola mundo
```

## Target decoder

Frase en español sin primer token:

```text
hola mundo <eos>
```

---

# 20. Tokens `<bos>` y `<eos>`

| Token | Significado | Función |
|---|---|---|
| `<bos>` | Beginning of sequence | Inicio de generación |
| `<eos>` | End of sequence | Final de generación |

Ejemplo:

```python
sentences_spanish = [
    f"<bos> {sent} <eos>"
    for sent in sentences_spanish
]
```

---

# 21. Decoder con cross-attention

El decoder tiene dos atenciones:

## 1. Masked self-attention

Mira lo generado hasta ahora.

```python
attn1 = MultiHeadAttention(...)(
    query=inputs,
    value=inputs,
    key=inputs,
    use_causal_mask=True
)
```

## 2. Cross-attention

Mira la salida del encoder.

```python
attn2 = MultiHeadAttention(...)(
    query=out1,
    value=encoder_output,
    key=encoder_output
)
```

---

# 22. Fine-tuning con BERT

Para clasificación de texto con Hugging Face:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=2
)
```

---

# 23. Tokenizar dataset con Hugging Face

```python
def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

encoded_data = dataset.map(tokenize, batched=True)
```

Formato PyTorch:

```python
encoded_data.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)
```

---

# 24. TrainingArguments

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./outputs",
    report_to="none",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no"
)
```

Si da error por versión antigua:

```python
evaluation_strategy="epoch"
```

en vez de:

```python
eval_strategy="epoch"
```

---

# 25. Trainer

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics
)
```

Entrenar:

```python
trainer.train()
```

Evaluar:

```python
trainer.evaluate()
```

---

# 26. Métricas

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    y_true = pred.label_ids
    y_pred = pred.predictions.argmax(-1)

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }
```

---

# 27. Inferencia sobre test

```python
test_predictions = trainer.predict(test_dataset)

logits = test_predictions.predictions
y_true = test_predictions.label_ids

probabilities = torch.nn.functional.softmax(
    torch.tensor(logits),
    dim=-1
).numpy()

y_pred = np.argmax(probabilities, axis=-1)
```

---

# 28. Inferencia sobre textos nuevos

```python
def predict_sentiment(text):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1).numpy()[0]

    pred_id = np.argmax(probs)

    return pred_id, probs
```

---

# 29. Modelos BERT del listado

| Modelo | Cuándo usarlo |
|---|---|
| `bert-base-uncased` | Inglés general, sin importar mayúsculas |
| `bert-base-cased` | Cuando las mayúsculas importan |
| `distilbert-base-uncased` | Más rápido y ligero |
| `albert-base-v2` | Menos parámetros, arquitectura optimizada |
| `xlm-roberta-base` | Textos multilingües |

---

# 30. SBERT

SBERT genera embeddings de frases directamente.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(sentences)
```

Modelos típicos:

| Modelo | Ventaja |
|---|---|
| `all-MiniLM-L6-v2` | Rápido y ligero |
| `all-mpnet-base-v2` | Mejor calidad, más pesado |

---

# 31. Similitud coseno

Sirve para comparar vectores.

```python
from sklearn.metrics.pairwise import cosine_similarity

sim = cosine_similarity(
    vec1.reshape(1, -1),
    vec2.reshape(1, -1)
)[0][0]
```

Interpretación:

| Valor | Significado |
|---|---|
| Cerca de 1 | Muy similares |
| Cerca de 0 | Poco relacionados |
| Negativo | Muy diferentes/opuestos |

---

# 32. BERT embeddings con `last_hidden_state`

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

tokens = tokenizer(
    text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=64
)

outputs = model(**tokens)

hidden_states = outputs.last_hidden_state
```

Forma:

```text
(batch_size, sequence_length, hidden_size)
```

En BERT base:

```text
hidden_size = 768
```

---

# 33. Mean pooling con BERT

```python
def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1)

    masked_embeddings = last_hidden_state * mask

    sum_embeddings = masked_embeddings.sum(dim=1)
    sum_mask = mask.sum(dim=1)

    return sum_embeddings / sum_mask
```

Versión sencilla eliminando tokens especiales:

```python
real_embeddings = []

for token, embedding in zip(tokens, hidden_states):
    if token not in ["[CLS]", "[SEP]", "[PAD]"]:
        real_embeddings.append(embedding)

sentence_embedding = np.mean(real_embeddings, axis=0)
```

---

# 34. Max pooling con BERT

```python
sentence_embedding = np.max(real_embeddings, axis=0)
```

| Pooling | Ventaja |
|---|---|
| Mean pooling | Más estable |
| Max pooling | Destaca señales fuertes |

---

# 35. Word2Vec vs BERT vs SBERT

| Modelo | Tipo | Ventaja | Limitación |
|---|---|---|---|
| Word2Vec | Estático | Simple y rápido | No entiende contexto |
| BERT | Contextual token-level | Entiende contexto | Necesita pooling para frase |
| SBERT | Contextual sentence-level | Ideal para similitud | Depende del modelo usado |

Para similitud entre frases:

```text
SBERT > BERT con pooling > Word2Vec promedio
```

---

# 36. Tokenizadores de dominio

## BERT general

```python
bert-base-uncased
```

Bueno para inglés general.

## PubMedBERT

```python
microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
```

Bueno para texto biomédico.

Ejemplo:

```python
bert_tokenizer.tokenize("adenocarcinoma")
pubmed_tokenizer.tokenize("adenocarcinoma")
```

Un tokenizador especializado suele fragmentar menos palabras técnicas.

---

# 37. Código base para comparar tokenizadores

```python
from transformers import AutoTokenizer

bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
pubmed_tok = AutoTokenizer.from_pretrained(
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)

term = "Immunohistochemistry"

print(bert_tok.tokenize(term))
print(pubmed_tok.tokenize(term))
```

---

# 38. Plantilla rápida: clasificación con BERT

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

dataset = load_dataset("rotten_tomatoes")

model_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

encoded = dataset.map(tokenize, batched=True)

encoded.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=2
)

args = TrainingArguments(
    output_dir="./outputs",
    report_to="none",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["validation"]
)

trainer.train()
trainer.evaluate()
```

---

# 39. Plantilla rápida: embeddings con SBERT

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

query = "Dogs are domestic animals."
sentences = [
    "Dogs are pets.",
    "This is a dog.",
    "They are free today."
]

q_emb = model.encode([query])
s_emb = model.encode(sentences)

scores = cosine_similarity(q_emb, s_emb)[0]

for sentence, score in zip(sentences, scores):
    print(sentence, score)
```

---

# 40. Preguntas típicas de examen

## ¿Qué diferencia hay entre embedding estático y contextual?

Estático:

```text
Una palabra = un vector fijo
```

Contextual:

```text
Una palabra = vector distinto según la frase
```

---

## ¿Por qué BERT necesita `[CLS]`?

Porque en clasificación se usa como representación global de la secuencia.

---

## ¿Qué hace `[SEP]`?

Marca el final de una secuencia o separa dos frases.

---

## ¿Qué hace `[PAD]`?

Rellena secuencias para que tengan la misma longitud.

---

## ¿Qué es la `attention_mask`?

Indica qué tokens son reales y cuáles son padding.

```text
1 = atender
0 = ignorar
```

---

## ¿Por qué se usa `truncation=True`?

Para evitar que una secuencia supere la longitud máxima permitida.

---

## ¿Qué es `use_causal_mask=True`?

Impide que el decoder mire tokens futuros.

Se usa en generación autoregresiva.

---

## ¿Cuándo uso encoder?

Cuando quiero clasificar o entender un texto.

---

## ¿Cuándo uso decoder?

Cuando quiero generar texto token a token.

---

## ¿Cuándo uso encoder-decoder?

Cuando quiero transformar una secuencia en otra.

Ejemplo:

```text
traducción
```

---

## ¿Por qué SBERT es mejor que BERT para similitud de frases?

Porque SBERT está entrenado específicamente para producir embeddings de frases comparables mediante similitud coseno.

---

# 41. Errores típicos

## Error 1: olvidarse de `truncation=True`

```python
tokenizer(text, max_length=128)
```

Mejor:

```python
tokenizer(text, max_length=128, truncation=True)
```

---

## Error 2: usar etiquetas con `NaN`

Antes de `train_test_split`:

```python
print(pd.Series(y).isna().sum())
```

---

## Error 3: no convertir textos a string

```python
df["Text"] = df["Text"].fillna("").astype(str)
```

---

## Error 4: no usar `attention_mask`

En BERT siempre conviene pasar:

```python
input_ids
attention_mask
```

---

## Error 5: comparar BERT token-level como si fuese sentence embedding

BERT devuelve embeddings por token.

Para frase necesitas pooling:

```text
mean pooling
max pooling
[CLS]
```

---

# 42. Resumen final ultra rápido

```text
Tokenización → input_ids + attention_mask
Embedding → texto convertido en vectores
Encoder → clasificación/comprensión
Decoder → generación
Encoder-decoder → traducción
BERT → embeddings contextualizados por token
SBERT → embeddings de frase
Word2Vec → embeddings estáticos
Padding → relleno
Truncation → recorte
[CLS] → clasificación
[SEP] → separador
[PAD] → relleno
Causal mask → no mirar futuro
Softmax → probabilidades
Argmax → clase predicha
Cosine similarity → similitud entre vectores
```

---

# 43. Mini chuleta de importaciones

```python
import numpy as np
import pandas as pd
import torch
import tensorflow as tf

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from datasets import load_dataset

from sentence_transformers import SentenceTransformer

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention
from tensorflow.keras.layers import LayerNormalization, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
```
