# 🖼️ PhotoMaster

A powerful image processing web app built with Python and Streamlit. PhotoMaster lets you batch-process product images from multiple sources — Excel sheets, Google Drive folders, PDFs, or direct uploads — with background removal, resizing, flipping, and ZIP export, all from a clean browser UI.

🔗 **Live Demo**: [photo-master.onshobbak.com](https://photo-master.onshobbak.com)

---

## ✨ Features

- **Multi-source input** — Upload images directly, or provide an Excel/CSV file with image URLs, embedded images, or a Google Drive folder link
- **Background removal** — AI-powered removal using the `briaai/RMBG-1.4` model via Hugging Face
- **Custom background** — Composite processed images onto any background (with optional foreground scaling)
- **Smart resizing** — Auto-resizes to 1024×1024 (or 1290×789 for banners), with configurable aspect-ratio cropping
- **PDF to image** — Converts each PDF page to a JPEG automatically
- **Image flipping** — Flip individual images horizontally or vertically
- **Inline rename** — Rename any image before downloading
- **Bulk download** — Download all processed images as a single ZIP file
- **Hyperlink extraction** — Reads actual hyperlink targets from Excel cells, not just cell text
- **Duplicate name handling** — Automatically suffixes duplicate names to avoid collisions

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| Image processing | [Pillow](https://python-pillow.org), [pypdfium2](https://github.com/pypdfium2-team/pypdfium2), [pdf2image](https://github.com/Belval/pdf2image) |
| AI background removal | [Hugging Face Transformers](https://huggingface.co) · `briaai/RMBG-1.4` |
| Data handling | [pandas](https://pandas.pydata.org), [openpyxl](https://openpyxl.readthedocs.io) |
| Google Drive API | [google-api-python-client](https://github.com/googleapis/google-api-python-client) |
| Web scraping | [BeautifulSoup4](https://beautiful-soup-4.readthedocs.io) |

---

## ⚙️ Installation

**Prerequisites:** Python 3.8+, pip

```bash
# 1. Clone the repository
git clone https://github.com/Abdelrhman-Ghowil/PhotoMaster.git
cd PhotoMaster

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run Final.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🐳 Docker Deployment

PhotoMaster is ready for Docker-based handoff and deployment. The Docker setup keeps large AI model files and secrets outside the image, so Ninja DevOps can mount persistent model storage and rotate credentials without rebuilding.

### Build Images

```bash
# CPU image, default deployment target
docker build --target cpu -t photomaster:cpu .

# NVIDIA GPU image, for hosts with NVIDIA Container Toolkit
docker build --target gpu -t photomaster:gpu .
```

### Run With Docker Compose

```bash
# Optional: create local deployment settings
cp .env.example .env

# Start the CPU service
docker compose up --build photomaster-cpu
```

Open [http://localhost:8501](http://localhost:8501).

For GPU deployments:

```bash
docker compose --profile gpu up --build photomaster-gpu
```

### Runtime Files And Secrets

- Mount or provide `assets/big-lama.pt` at `/app/assets/big-lama.pt`.
- Mount or provide `assets/models/RealESRGAN_x4plus.pth` at `/app/assets/models/RealESRGAN_x4plus.pth`.
- Provide the Google Drive service-account JSON outside Git and set `GOOGLE_CREDENTIALS_FILE` in `.env` if it is not `./credentials.json`.
- The container reads Google credentials from `GOOGLE_APPLICATION_CREDENTIALS`, defaulting to `/run/secrets/google_credentials`.
- Hugging Face and Torch caches are persisted through Docker volumes so model downloads are not repeated on every restart.

The app can still start without Google credentials; only Google Drive features require the service-account file. If model files are missing, the app will try to download supported models into the mounted `/app/assets` path or configured cache volumes.

### Production Notes

- Keep model storage persistent across deploys; first-run model downloads can be large and slow.
- Allocate enough memory for PyTorch, Transformers, image processing, and PDF conversion workloads.
- CPU is the default runtime. For GPU deployments, install and enable NVIDIA Container Toolkit on the host before using the `gpu` compose profile.
- The container healthcheck probes Streamlit at `/_stcore/health`.

### Ninja DevOps Handoff Checklist

- Provide Google service-account JSON outside Git.
- Preload or allow download of `assets/big-lama.pt`.
- Ensure `assets/models/RealESRGAN_x4plus.pth` is mounted or downloadable.
- Enable NVIDIA Container Toolkit before running the GPU profile.
- Confirm `http://localhost:8501` responds after deployment.

---

## 🔑 Google Drive Integration

To use the Google Drive folder feature, you need a service account:

1. Create a service account in [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the **Google Drive API**
3. Download the JSON key and save it as `credentials.json` in the project root, or set `GOOGLE_APPLICATION_CREDENTIALS` to a custom path
4. Share your Drive folder with the service account email

> ⚠️ **Never commit `credentials.json` to version control.** It is already listed in `.gitignore`.

---

## 🧽 Object Eraser Model

The Object Eraser feature uses a Big LaMa TorchScript model. Keep the model outside GitHub because it is a large binary file.

On the Shobbak server, upload the model to:

```bash
assets/big-lama.pt
```

Or set a custom path before starting Streamlit:

```bash
OBJECT_ERASER_MODEL_PATH=/home/shobbak/superpower/App_v1/assets/big-lama.pt
```

If the file is missing, the app can try to download it from `OBJECT_ERASER_MODEL_URL`.

Performance knobs:

```bash
OBJECT_ERASER_SIZE_LIMIT=1280
TORCH_NUM_THREADS=2
```

---

## 📋 Excel File Format

When uploading an Excel or CSV file with image links, the sheet must contain these two columns:

| name | links |
|---|---|
| product-001 | https://drive.google.com/... |
| product-002 | https://i.postimg.cc/... |

- The `name` column is used as the output filename
- The `links` column supports Google Drive share links, Postimg, ibb.co, and direct image URLs
- Embedded images in `.xlsx` files are also supported (select "Embedded in Excel file" when prompted)

---

## 📁 Project Structure

```
PhotoMaster/
├── Dockerfile          # CPU/GPU container build targets
├── docker-compose.yml  # Local and production compose example
├── .env.example        # Deployment environment template
├── Final.py            # Main Streamlit application
├── Bg.png              # Default background image
├── requirements.txt    # Python dependencies
├── credentials.json    # Google service account key (not committed)
└── README.md
```

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 👤 Author

**Abdelrhman Ghowil**
[GitHub](https://github.com/Abdelrhman-Ghowil) · [LinkedIn](https://linkedin.com/in/abdelrhman-ghowil/)
