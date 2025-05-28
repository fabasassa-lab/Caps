# Mango leaf Analyze (MANGALYZE) - LAI25-SM006

Proyek ini bertujuan untuk mengembangkan Mango Leaf Analyze (MANGALYZE), sebuah sistem deteksi kondisi daun mangga berbasis teknologi pengenalan citra dan logika rekomendasi sederhana. Latar belakang permasalahan muncul dari kesulitan yang dihadapi petani dalam mendeteksi dini penyakit atau gangguan pada daun mangga, yang sering kali berdampak pada penurunan hasil panen dan peningkatan biaya produksi. Sistem ini dirancang untuk mengklasifikasikan kondisi daun berdasarkan foto yang diunggah oleh pengguna, ke dalam beberapa kategori, seperti daun sehat, terserang penyakit (misalnya antraknosa atau bercak daun), serta gejala kekurangan nutrisi seperti klorosis. Proses klasifikasi dilakukan menggunakan teknik Convolutional Neural Network (CNN), yang telah terbukti efektif dalam pengenalan pola visual. Hasil klasifikasi kemudian diikuti oleh pemberian rekomendasi penanganan berbasis logika aturan (rule-based logic) yang sederhana dan mudah dipahami. Dengan pendekatan ini, MANGALYZE diharapkan mampu memberikan solusi praktis, cepat, dan akurat dalam pemantauan kesehatan daun mangga, serta mendukung implementasi pertanian musiman yang lebih efisien dan berkelanjutan. Sistem ini juga berpotensi menjadi kontribusi nyata dalam penerapan pertanian presisi dan pengelolaan sumber daya di sektor hortikultura lokal Indonesia.

## Anggota 
1. A296YBF008 – Achmad Fauzihan Bagus Sajiwo – Universitas Pembangunan Nasional Veteran Jawa Timur
2. A312XBF453 – Sevyra Nanda Octavianti – Universitas Sebelas Maret
3. A673XBF392 – Permata Ayu Rahmawati – Universitas Bhayangkara Surabaya
4. A308YBF407 – Rahmad Ramadhan Laska – Universitas Riau

### Setup Environment

```
# 1. Salin Repository dari GitHub
git clone https://github.com/fabasassa-lab/Capstone_Mangalyze.git

# 2. Pindah ke Folder Proyek
cd nama-repository
```

```
conda create --name main-ds python=3.10
conda activate main-ds
pip install flask keras tensorflow
```

### Run main.py

```
python main.py
```

link [model](https://drive.google.com/drive/folders/1YDotVPn0Fxy6ZGIgPMFUf6MiZOaR8FF_?usp=sharing)
