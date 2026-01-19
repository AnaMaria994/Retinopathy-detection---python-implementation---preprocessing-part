
# Detecția Retinopatiei Diabetice folosind ResNet-50

## 1. Prezentarea generală a proiectului

Acest proiect prezintă o soluție de diagnosticare bazată pe deep learning pentru  **detecția și clasificarea timpurie a Retinopatiei Diabetice (DR)**  – una dintre principalele cauze ale pierderii vederii în rândul pacienților cu diabet.

Folosind puterea  **ResNet-50**, o rețea neuronală convoluțională (CNN) pre-antrenată, modelul clasifică imaginile retiniene de înaltă rezoluție în cinci niveluri de severitate ale DR:

-   **0**  – Fără DR
    
-   **1**  – Ușoară (Mild)
    
-   **2**  – Moderată
    
-   **3**  – Severă
    
-   **4**  – DR Proliferativă
    

Detecția timpurie a DR este crucială pentru intervenția medicală la timp și prevenirea leziunilor ireversibile ale vederii. Acest proiect servește ca un instrument de suport decizional pentru oftalmologi și profesioniști din domeniul sănătății, oferind o clasificare rapidă și automată a imaginilor retiniene cu o acuratețe ridicată.

----------

##  2. Puncte cheie

-   **Model utilizat:**  ResNet-50 (pre-antrenat pe ImageNet)
    
-   **Frameworks & Instrumente:**  Python, TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn
    
-   **Tehnici:**  Preprocesare imagini, augmentare date, învățare prin transfer (transfer learning), evaluare AUC și acuratețe
    
-   **Performanță:**  **TBA**
    
-   **Set de date:**  Setul de date APTOS 2019 Blindness Detection de pe Kaggle (peste 35.000 de imagini)
    
-   **Tipul problemei:**  Clasificare de imagini multi-clasă (5 clase)
    

----------

## 3. Definirea problemei

Dezvoltarea unui model automat de deep learning care să poată detecta și clasifica cu precizie stadiul Retinopatiei Diabetice din imaginile fundului de ochi, minimizând timpul de screening manual și crescând eficiența diagnosticului.

----------

## 🔬 4. Analiza literaturii de specialitate (State-of-the-Art)

| Nr. | Autor(i) | An | Titlul articolului/proiectului | Aplicație sau Domeniu¹ | Tehnologii utilizate² | Metodologie sau Abordare³ | Rezultate⁴ | Limitări⁵ | Comentarii suplimentare⁶ |
|:---:|:---|:---:|:---|:---|:---|:---|:---|:---|:---|
| 1 | **Surya vamsi Patiballa** | ~2023 | [Detecția Retinopatiei Diabetice folosind ResNet-50](https://www.linkedin.com/in/surya-patiballa-b724851aa/) | Clasificare multi-clasă DR (5 stadii) | Python, TensorFlow, Keras, ResNet-50 | Transfer Learning (ResNet-50 pre-antrenat) pe setul de date APTOS 2019. Augmentare cu `ImageDataGenerator`. | Validare AUC: 94%<br>Training AUC: 97.77%<br>Training Acc: 87% | Suprapotrivire (overfitting) ușoară (Train AUC > Validare AUC). | **Proiectul curent.** O implementare solidă a ResNet-50. |
| 2 | Karthika, S., et al. | 2024 | [Enhancing Diabetic Retinopathy Diagnosis with ResNet-50-Based Transfer Learning](https://ideas.repec.org/a/spr/aodasc/v11y2024i1d10.1007_s40745-023-00494-0.html) | Clasificare DR (5 stadii) | ResNet-50 | Transfer Learning cu ResNet-50. Preprocesare și segmentare, urmate de înghețarea unor straturi și Global Average Pooling. | Acuratețe: 99.82%<br>Sensibilitate: 99%<br>Specificitate: 96%<br>AUC: 0.99 | Evaluat pe APTOS-2019 și un set de date mic (40 imagini) în timp real. | Rezultate excepționale. Metoda de preprocesare pare a fi cheia. |
| 3 | Patra, P. & Singh, T. | 2022 | [Diabetic Retinopathy Detection using an Improved ResNet50-InceptionV3 Structure](https://www.semanticscholar.org/paper/Diabetic-Retinopathy-Detection-using-an-Improved-Patra-Singh/e9dd4cd8ea15d6c374c7e55a9392e772abc3761f) | Clasificare DR (5 stadii) | ResNet-50, InceptionV3, CNN | Abordare hibridă care combină ResNet-50 și InceptionV3 pentru extragerea trăsăturilor. | Acuratețe: 83.79% | Acuratețea este mai mică decât a altor modele, sugerând că hibridizarea nu a fost optimă. | Compară cu îmbunătățirile viitoare (VGG, Inception). Acest articol deja le combină. |
| 4 | Wu, et al. | 2023 | [Development of revised ResNet-50 for diabetic retinopathy detection](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-023-05293-1) | Clasificare/Gradare DR | ResNet-50, Python | O structură ResNet-50 "revizuită", cu regularizare și rată de învățare adaptivă pentru a evita overfitting-ul. | Train Acc: 83.95%<br>Test Acc: 74.32% | Acuratețe de testare relativ scăzută, dar evită overfitting-ul. | Relevant pentru problema overfitting-ului observată în proiectul de bază. |
| 5 | Putu Gede Yoga Pramana Putra | 2025 | [Comparison of ResNet-50 and DenseNet-121 Architectures in Classifying Diabetic Retinopathy](https://jurnal.yoctobrain.org/index.php/ijodas/article/download/232/223/) | Clasificare DR (multi-clasă) | ResNet-50, DenseNet-121, K-Fold Cross Validation | Comparație directă între ResNet-50 și DenseNet-121, folosind validare încrucișată K-Fold. | ResNet-50 a depășit DenseNet-121 (metricile exacte nu sunt în snippet). | Set de date limitat (2000 imagini), ceea ce afectează generalizarea. | Confirmă alegerea ResNet-50 ca fiind o arhitectură robustă for această sarcină. |

----------

##  5. Workflow
1.  **Achiziția datelor:**  Imagini de înaltă rezoluție ale fundului de ochi de pe Kaggle.
    
2.  **Preprocesarea datelor:**
    
    -   Redimensionarea imaginilor (224x224)
        
    -   Normalizare și eliminarea zgomotului
        
    -   Augmentarea datelor folosind  `ImageDataGenerator`
        
3.  **Arhitectura modelului:**
    
    -   Extragerea trăsăturilor cu ResNet-50
        
    -   Fine-tuning (ajustare fină) a capului de clasificare
        
4.  **Antrenare:**
    
    -   Optimizator: Adam
        
    -   Funcție de pierdere (Loss): Categorical Crossentropy
        
    -   Epoci: 100
        
5.  **Evaluare:**
    
    -   Acuratețe
        
    -   Loss
        
    -   Scor AUC
        
    -   Matrice de confuzie
        

----------

##  6. Rezultate

TBA

----------

##  7. Structura fișierelor

```
.
├── DR.py                        # Scriptul principal Python cu implementarea ResNet-50
├── trainLabels.csv             # Etichete adnotate pentru antrenare
├── /Dataset/                   # Director cu imagini retiniene
├── README.md                   # Acest fișier
└── [Grafice suplimentare & artefacte ale modelului]

```

----------

##  8. Autor și Mentor

### Mentor

Acest proiect a fost finalizat pentru materia Procesarea Imaginilor    **Universitatea Tehnică Gheorghe Asachi**, sub îndrumarea domnului  **Achirei Ștefan Daniel**, ș.l. dr.inginer.

### Autor

**Ana-Maria Panaite**  Calculatoare și Tehnologia Informației- Facultatea de Automatică și Calculatoare din Iași

-   **Email:**  ana-maria.panainte@student.tuiasi.ro	
    
-   **LinkedIn:**  [https://www.linkedin.com/in/ana-maria-panaite]        
**George Cătănescu**  Calculatoare și Tehnologia Informației- Facultatea de Automatică și Calculatoare din Iași

-   **Email:**  george.catanescu@student.tuiasi.ro	
    
-   **LinkedIn:**  [https://www.linkedin.com/in/george-catanescu]
----------
