from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path
import torch
import io
from PIL import Image
import torchvision.transforms as T

APP_ROOT = Path(__file__).parent
ARTIFACT_PATH = APP_ROOT / 'artifacts' / 'detector.pt'

app = FastAPI(title='Adversarial Deepfake Detector Demo')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

app.mount('/static', StaticFiles(directory=APP_ROOT / 'static'), name='static')


def load_model():
    if ARTIFACT_PATH.exists():
        try:
            model = torch.jit.load(str(ARTIFACT_PATH))
            model.eval()
            return model, 'torchscript'
        except Exception as e:
            print('Failed to load TorchScript artifact:', e)
    # fallback
    from models import SmallCNN
    model = SmallCNN()
    model.eval()
    return model, 'smallcnn-fallback'


MODEL, MODEL_TYPE = load_model()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.get('/')
def index():
    index_path = APP_ROOT / 'static' / 'index.html'
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse('<h1>Adversarial Deepfake Detector Demo</h1>')


@app.get('/health')
def health():
    return {'status': 'ok', 'model': MODEL_TYPE}


def predict_tensor(img_tensor: torch.Tensor):
    with torch.no_grad():
        out = MODEL(img_tensor.unsqueeze(0))
        probs = torch.softmax(out, dim=1)
        # Simple smoke rule: if max prob < 0.6 => suspicious (low confidence)
        maxp, cls = probs.max(dim=1)
        verdict = 'not_attacked' if maxp.item() >= 0.6 else 'attacked'
        return {
            'verdict': verdict,
            'confidence': float(maxp.item()),
            'pred_class': int(cls.item())
        }


@app.post('/upload')
async def upload_image(file: UploadFile = File(...)):
    content = await file.read()
    try:
        img = Image.open(io.BytesIO(content)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid image')
    img_t = transform(img)
    result = predict_tensor(img_t)
    return JSONResponse(result)


@app.get('/demo')
def demo():
    # Return minimal demo metadata; front-end will fetch demo images from static/
    return {
        'demo_images': {
            'clean': '/static/demo_clean.png',
            'attacked': '/static/demo_attacked.png'
        },
        'note': 'The attacked image demonstrates a misclassification on a victim model (demo only).'
    }
