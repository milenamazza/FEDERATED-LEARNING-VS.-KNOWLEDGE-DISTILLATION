# Federated Learning vs. Knowledge Distillation
## Obiettivo
Il progetto affronta il problema della classificazione automatica di routine fisioterapiche in scenari distribuiti, dove i dati dei pazienti:

- Sono raccolti localmente presso diverse strutture sanitarie
- Non possono essere centralizzati per motivi di privacy e vincoli normativi
- Presentano forte eterogeneitĂ  statistica (distribuzioni non-IID)
- Sono caratterizzati da sbilanciamento tra classi

## Approccio
Il lavoro confronta tre paradigmi di apprendimento distribuito:

- Federated Learning Classico: i client addestrano localmente copie del modello globale e condividono solo i pesi aggregati (FedAvg, FedProx, FedAdam, FedAdagrad)
- Knowledge Distillation Centralizzata: un modello teacher centralizzato trasferisce conoscenza a modelli student locali tramite soft labels
- Knowledge Distillation Federata: i client collaborano condividendo predizioni su un dataset sintetico comune (FedMD, FedKD)



## 📂 Struttura del progetto

- **`femnist_lab/`**
  - Codice per esperimenti di *Federated Learning* sul dataset **FEMNIST** [implementazioni FedAvg, FedProx, FedAdam, FedAdagrad].
- **`irds_lab/`**
  - Implementazioni di *Federated Learning* sul dataset **IRDS** [implementazioni FedAvg, FedProx, FedAdam, FedAdagrad].
- **`kd_centralizzata/`**
  - Codice per la **Knowledge Distillation centralizzata** [implementazioni teacher-student su FEMNIST e IRDS].
- **`knwledge_distilaltion_federata/`**
  - Implementazioni di **Knowledge Distillation federata** [implementazioni FedMD e FedKD].


## ⚙️ Requisiti

- Python 3.9+
- Librerie principali:
  - `torch`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `flwr`  ← **Framework per il Federated Learning**


