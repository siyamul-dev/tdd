from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# origins = [
#     "http://localhost",
#     "http://localhost:3000",
#     "http://tdd.siyamul.com",
#     "https://tdd.siyamul.com",
# ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("models/1")

CLASS_NAMES = ['Bacterial_spot',
 'Early_blight',
 'Healthy',
 'Late_blight',
 'Leaf_Mold',
 'Mosaic_virus',
 'Septoria_leaf_spot',
 'YellowLeaf_Curl_Virus']

@app.get("/ping")
async def ping():
Expand All
	@@ -46,8 +48,12 @@ async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
Expand All
	@@ -58,5 +64,5 @@ async def predict(
    }

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=10000)
