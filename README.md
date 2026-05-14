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

## 🔑 Google Drive Integration

To use the Google Drive folder feature, you need a service account:

1. Create a service account in [Google Cloud Console](https://console.cloud.google.com/)
2. Enable the **Google Drive API**
3. Download the JSON key and save it as `credentials.json` in the project root
4. Share your Drive folder with the service account email

> ⚠️ **Never commit `credentials.json` to version control.** It is already listed in `.gitignore`.

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
