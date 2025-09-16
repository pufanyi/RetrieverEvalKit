# !pip install transformers>=4.52.0 torch>=2.6.0 peft>=0.15.2 torchvision pillow
# !pip install
import torch
from transformers import AutoModel

# Initialize the model
model = AutoModel.from_pretrained(
    "jinaai/jina-embeddings-v4", trust_remote_code=True, torch_dtype=torch.float16
)

model.to("cuda")

# ========================
# 1. Retrieval Task
# ========================
# Configure truncate_dim, max_length (for texts), max_pixels (for images), vector_type,
# batch_size in the encode function if needed

# Encode query
query_embeddings = model.encode_text(
    texts=["Overview of climate change impacts on coastal cities"],
    task="retrieval",
    prompt_name="query",
)

# Encode passage (text)
passage_embeddings = model.encode_text(
    texts=[
        "Climate change has led to rising sea levels,"
        "increased frequency of extreme weather events..."
    ],
    task="retrieval",
    prompt_name="passage",
)

# Encode image/document
image_embeddings = model.encode_image(
    images=["https://i.ibb.co/nQNGqL0/beach1.jpg"],
    task="retrieval",
)

# ========================
# 2. Text Matching Task
# ========================
texts = [
    "غروب جميل على الشاطئ",  # Arabic
    "海滩上美丽的日落",  # Chinese
    "Un beau coucher de soleil sur la plage",  # French
    "Ein wunderschöner Sonnenuntergang am Strand",  # German
    "Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία",  # Greek
    "समुद्र तट पर एक खूबसूरत सूर्यास्त",  # Hindi
    "Un bellissimo tramonto sulla spiaggia",  # Italian
    "浜辺に沈む美しい夕日",  # Japanese
    "해변 위로 아름다운 일몰",  # Korean
]

text_embeddings = model.encode_text(texts=texts, task="text-matching")

# ========================
# 3. Code Understanding Task
# ========================

# Encode query
query_embedding = model.encode_text(
    texts=["Find a function that prints a greeting message to the console"],
    task="code",
    prompt_name="query",
)

# Encode code
code_embeddings = model.encode_text(
    texts=["def hello_world():\n    print('Hello, World!')"],
    task="code",
    prompt_name="passage",
)

# ========================
# 4. Use multivectors
# ========================

multivector_embeddings = model.encode_text(
    texts=texts,
    task="retrieval",
    prompt_name="query",
    return_multivector=True,
)

images = ["https://i.ibb.co/nQNGqL0/beach1.jpg", "https://i.ibb.co/r5w8hG8/beach2.jpg"]
multivector_image_embeddings = model.encode_image(
    images=images,
    task="retrieval",
    return_multivector=True,
)
