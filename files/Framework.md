Based on my comprehensive search, I'll now provide you with a detailed compilation of papers on aircraft maneuver detection with a focus on unsupervised learning methods.

# Papers on Aircraft Maneuver Detection: Focus on Unsupervised Learning

## Overview

Aircraft maneuver detection is a critical area of research in aviation safety, flight analysis, and autonomous systems. While supervised learning methods are widely used, unsupervised approaches offer unique advantages including the ability to detect unknown anomalies, operate without labeled data, and identify novel patterns. This compilation covers key papers in the field with particular emphasis on unsupervised learning techniques.

## Core Unsupervised Learning Approaches

### Autoencoder-Based Methods

**Convolutional Variational Auto-Encoder (CVAE)** represents one of the most promising approaches. NASA researchers developed an unsupervised CVAE model for anomaly detection in high-dimensional heterogeneous time-series flight data. The model demonstrated superior performance compared to supervised methods and traditional unsupervised approaches like K-Means++ and One-Class SVM on Yahoo's benchmark datasets. The CVAE approach was successfully applied to identify anomalies in the first 60 seconds of commercial flights' take-offs using Flight Operational Quality Assurance (FOQA) data.[1]

**AutoEncoder-LSTM frameworks** have emerged as powerful tools for real-time anomaly detection. A notable implementation developed for COMAC C919 flight testing combines a fine-tuned autoencoder for feature extraction with stacked LSTM networks for temporal pattern recognition. This approach addresses the curse of dimensionality while maintaining real-time processing capabilities, achieving significant performance improvements over traditional methods.[2]

**Bi-LSTM Autoencoders** have shown exceptional capability in satellite maneuver detection, with applications extending to aircraft systems. These models learn to reconstruct normal flight patterns and flag deviations as potential maneuvers or anomalies. The reconstruction error serves as an anomaly score, with higher errors indicating more likely maneuvers.[3]

### Clustering-Based Approaches

**DBSCAN (Density-Based Spatial Clustering)** has proven effective for trajectory analysis and anomaly detection. Research shows DBSCAN can automatically identify noise data and outliers in flight trajectories, making it particularly valuable for unsupervised maneuver detection. The algorithm's ability to discover clusters of arbitrary shapes makes it well-suited for complex flight patterns.[4][5]

**K-Means clustering** combined with Gaussian Mixture Models (GMM) has been successfully applied to aircraft trajectory recognition. A notable implementation developed an incremental GMM-based anomaly detection method that continuously adapts to new incoming flight data, addressing the limitation of offline-only training approaches.[6][7]

**Spectral clustering** methods have been applied to terminal area operations, using Wasserstein-distance-based algorithms to divide historical trajectories into performance-based groups. This approach is particularly effective for identifying distinct flight patterns and operational modes.[8]

### One-Class Classification Methods

**One-Class Support Vector Machines (OC-SVM)** have been extensively studied for flight anomaly detection. These models learn decision boundaries around normal flight operations, identifying anything outside these boundaries as anomalous. Research shows OC-SVM can be effective for detecting outliers in flight data, though performance varies depending on the specific application and data characteristics.[9][10]

**Isolation Forest** algorithms have demonstrated promising results in flight anomaly detection systems. The method's ability to isolate anomalies using fewer partitions makes it computationally efficient and well-suited for high-volume flight data processing. GitHub implementations show good precision, recall, and F1-score performance for both normal and anomalous flight patterns.[11]

## Advanced Deep Learning Approaches

### Transformer-Based Models

**Attention mechanisms** in transformer architectures have been applied to aircraft maneuver detection with notable success. These models can capture long-range dependencies in flight data and have shown superior performance in trajectory generation and anomaly detection tasks. A missile flight data study demonstrated that transformer-based encoders with multi-head attention significantly outperform traditional machine learning approaches.[12][13]

**Position-sensitive self-attention** units have been developed to address limitations in standard transformer architectures for aircraft engine RUL prediction, with applications extending to maneuver detection. These enhanced transformers better incorporate local context while maintaining global pattern recognition capabilities.[14]

### Generative Adversarial Networks (GANs)

**GAN-based anomaly detection** approaches have been explored for complex multivariate time series analysis in aerospace applications. While primarily used for representation learning rather than data augmentation, GANs show promise for learning normal flight patterns and detecting deviations. LSTM-RNN implementations within GAN frameworks have demonstrated effectiveness in capturing temporal dependencies in flight data.[15][16]

### Variational Approaches

**Variational Autoencoders (VAEs)** have been successfully applied to loss-of-control detection in commercial aircraft. These models use conditional frameworks to capture belief states and detect anomalous flight conditions through reconstruction probability and latent space analysis. NASA research on the GTM T-2 aircraft confirmed the effectiveness of VAE-based approaches for identifying loss-of-control events.[17]

## Statistical and Traditional Approaches

### Hidden Markov Models (HMMs)

**Hidden Markov Models** have been extensively studied for pilot behavior analysis and flight phase identification. While traditionally used for understanding pilot scanning patterns, HMMs have been adapted for automated maneuver detection by modeling the hidden states corresponding to different flight maneuvers. Recent implementations achieve 97% accuracy in identifying flight phases including taxi, takeoff, climb, cruise, approach, and rollout.[18][19][20]

### Kernel Density Estimation

**Functional Kernel Density Estimation** approaches have been developed for time series anomaly detection in aviation data. These methods extend traditional KDE to Hilbert spaces, offering both point and Fourier-based approaches for scoring time series anomalies. The methods naturally handle missing data and demonstrate competitive performance with other functional data analysis techniques.[21]

### Gaussian Mixture Models

**Gaussian Mixture Model clustering** has been widely adopted for flight operation analysis. These probabilistic models can identify common flight patterns and detect outliers through likelihood estimation. Incremental GMM implementations address the challenge of continuously growing flight data by updating clusters rather than retraining from scratch.[7][6]

## Specialized Applications

### Satellite Maneuver Detection

**Unsupervised satellite maneuver detection** using autoencoder approaches has shown exceptional results. These methods analyze orbital parameters from Two Line Elements (TLEs) to detect anomalous behavior indicating maneuvers. Bi-LSTM autoencoders trained on debris data (assumed purely ballistic) effectively detect maneuvers in operational satellites as deviations from expected patterns.[22][3]

### Air Combat Maneuvers

**Time series segmentation and clustering** approaches have been developed specifically for air combat maneuver pattern extraction. These methods combine autoencoder feature extraction with clustering analysis to identify and classify complex combat maneuvers from flight data.[23]

### Real-Time Processing

**FPGA-based implementations** have been developed for real-time autoencoder-LSTM processing in flight testing applications. These hardware-accelerated systems achieve significant speed improvements (36.3x faster than CPU, 23.9x faster than GPU) while consuming substantially less energy, making them viable for embedded flight systems.[2]

## Evaluation and Validation

### Performance Metrics

Research consistently shows that unsupervised methods can achieve high accuracy rates. Notable results include:
- CVAE models achieving superior performance compared to traditional methods on benchmark datasets[1]
- Autoencoder-LSTM frameworks reaching 93% precision in commercial aircraft anomaly detection[24]
- Hidden Markov Models achieving 97% accuracy in flight phase identification[20]
- Transformer-based models outperforming traditional approaches in multiple aerospace applications[13]

### Datasets and Validation

Common datasets used for validation include:
- **FOQA (Flight Operational Quality Assurance)** data for commercial aviation analysis[1]
- **ADS-B surveillance data** for trajectory analysis and anomaly detection[25][26]
- **NASA datasets** including GTM T-2 aircraft data for loss-of-control studies[17]
- **C-MAPSS dataset** for engine health monitoring and anomaly detection[27]

## Current Challenges and Future Directions

### Data Quality and Availability

Unsupervised methods face challenges with data quality, particularly when dealing with TLE data that contains inherent errors and noise. Future developments focus on improving robustness to input data quality and developing methods that can handle incomplete or corrupted flight data.[3]

### Real-Time Processing

While significant progress has been made in developing real-time capable systems, balancing accuracy with computational efficiency remains an active area of research. Hardware acceleration approaches show promise for meeting stringent timing requirements in operational systems.

### Interpretability

Enhancing the interpretability of unsupervised models remains crucial for adoption in safety-critical aviation applications. Research is ongoing to develop methods that not only detect maneuvers but also provide meaningful explanations for their decisions.

This comprehensive overview demonstrates the rich variety of unsupervised learning approaches being applied to aircraft maneuver detection, with promising results across multiple domains and applications. The field continues to evolve with advances in deep learning, hardware acceleration, and specialized aerospace applications.

[1](https://ntrs.nasa.gov/api/citations/20200011471/downloads/20200011471.pdf)
[2](https://www.doc.ic.ac.uk/~wl/papers/19/fpt19zq.pdf)
[3](https://conference.sdo.esoc.esa.int/proceedings/sdc9/paper/249/SDC9-paper249.pdf)
[4](https://www.extrica.com/article/21582/pdf)
[5](https://arc.aiaa.org/doi/10.2514/6.2020-1851)
[6](https://arxiv.org/abs/2005.09874)
[7](https://www.sciencedirect.com/science/article/abs/pii/S0968090X16000188)
[8](https://www.sciencedirect.com/science/article/abs/pii/S1270963822006848)
[9](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12058/120583A/Flight-operation-anomaly-detection-based-on-one-class-SVM/10.1117/12.2619663.full)
[10](https://www.ijert.org/development-of-anomaly-detection-system-for-flight-data-using-ai)
[11](https://github.com/AQk1/Flight-Anomaly-Detection-and-Prediction)
[12](https://ntrs.nasa.gov/api/citations/20250006116/downloads/v4.pdf)
[13](https://pure.kaist.ac.kr/en/publications/anomaly-detection-method-for-missile-flight-data-by-attention-cnn)
[14](https://www.nature.com/articles/s41598-024-59095-3)
[15](https://arxiv.org/pdf/2110.12076.pdf)
[16](https://arxiv.org/pdf/1809.04758.pdf)
[17](https://ntrs.nasa.gov/api/citations/20205009996/downloads/CampbellGrauer_SciTech_2021%20-%20Intelligent_Systems_v4.pdf)
[18](https://dspace.mit.edu/bitstream/handle/1721.1/28912/60495284-MIT.pdf;sequence=2)
[19](https://rosap.ntl.bts.gov/view/dot/9124/dot_9124_DS1.pdf)
[20](https://journals.open.tudelft.nl/joas/article/view/7269/6092)
[21](https://pmc.ncbi.nlm.nih.gov/articles/PMC7759980/)
[22](https://www.politesi.polimi.it/retrieve/d5419fe2-383b-485c-9a75-9532db1d9287/2025_04_Raviola_Executive%20Summary.pdf)
[23](https://www.sciencedirect.com/science/article/pii/S2214914723002982)
[24](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0317914)
[25](https://arc.aiaa.org/doi/10.2514/6.2023-4107)
[26](https://www.sesarju.eu/sites/default/files/documents/sid/2018/papers/SIDs_2018_paper_17.pdf)
[27](https://arxiv.org/html/2502.05428v1)
[28](https://ntrs.nasa.gov/api/citations/20030014137/downloads/20030014137.pdf)
[29](https://www.aimspress.com/article/doi/10.3934/era.2023005?viewType=HTML)
[30](https://dl.acm.org/doi/abs/10.3233/JCM-204511)
[31](https://saemobilus.sae.org/papers/automatic-maneuver-detection-flight-data-using-wavelet-transform-deep-learning-algorithms-2024-26-0462)
[32](https://corescholar.libraries.wright.edu/cgi/viewcontent.cgi?article=1018&context=isap_2025)
[33](https://www.icck.org/filebob/uploads/storage/CJIF_d8vLQH5gJbtNmSM.pdf)
[34](https://www.ama-science.org/proceedings/download/AQVlAD==)
[35](http://liu.diva-portal.org/smash/get/diva2:1432871/FULLTEXT01.pdf)
[36](https://c3.ndc.nasa.gov/dashlink/static/media/publication/Trajectory_Clustering-dashlink.pdf)
[37](https://researchportal.lsbu.ac.uk/files/6335583/621925bf_Intelligent_20Flight_20Control_20of_20Combat_20Aircraft_20Based_20on_20Autoencoder.docx)
[38](https://www.sesarju.eu/sites/default/files/documents/sid/2017/SIDs_2017_paper_45.pdf)
[39](https://www.sciencedirect.com/science/article/abs/pii/S1270963825001166)
[40](https://www.ijraset.com/research-paper/anomaly-detection-in-time-series-flight-parameter-data-using-machine-learning-approach)
[41](https://www.sciencedirect.com/science/article/pii/S2772662223000619)
[42](https://dl.acm.org/doi/10.1145/3351180.3351210)
[43](https://www.sciencedirect.com/science/article/abs/pii/S1568494625003345)
[44](https://www.aanda.org/articles/aa/full_html/2017/10/aa30968-17/aa30968-17.html)
[45](https://en.wikipedia.org/wiki/Isolation_forest)
[46](https://ceur-ws.org/Vol-2289/paper12.pdf)
[47](https://www.sciencedirect.com/science/article/abs/pii/S0167865524003118)
[48](https://arxiv.org/html/2403.10802v1)
[49](https://www.sciencedirect.com/science/article/pii/S2214914725002405)
[50](https://www.diva-portal.org/smash/get/diva2:1764897/FULLTEXT01.pdf)
[51](https://arc.aiaa.org/doi/10.2514/1.I011246)
[52](https://www.sciencedirect.com/science/article/pii/S2666827022001219)
[53](https://www.sciencedirect.com/science/article/abs/pii/S1367578821000778)
[54](https://www.sciencedirect.com/science/article/pii/S0924271625003089)
[55](https://www.icck.org/article/abs/cjif.2024.344084)
[56](https://pubmed.ncbi.nlm.nih.gov/24963338/)
[57](https://arxiv.org/pdf/1710.01925.pdf)
[58](https://pmc.ncbi.nlm.nih.gov/articles/PMC9615909/)
[59](https://gilab.udg.edu/wp-content/uploads/publications/IIiA-15-01-RR.pdf)
[60](https://arxiv.org/abs/2107.13108)
[61](https://amostech.com/TechnicalPapers/2023/Poster/Kato.pdf)
[62](https://arxiv.org/abs/2112.03765)
[63](https://icact.org/upload/2020/0257/20200257_finalpaper.pdf)
[64](https://www.sciencedirect.com/science/article/abs/pii/S1570870521000433)
[65](https://www.sciencedirect.com/science/article/abs/pii/S0959652619306407)
[66](https://asmedigitalcollection.asme.org/risk/article/11/1/011107/1207852/Daily-Engine-Performance-Trending-Using-Common)