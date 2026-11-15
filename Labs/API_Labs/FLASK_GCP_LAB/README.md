# Flask + Streamlit on Google Cloud Run

Watch the tutorial video at [Tutorial Video](https://www.youtube.com/watch?v=O0X6NoQyEf0)

## Setup

Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) for your operating system and make sure the `gcloud` command works.

```
gcloud init
```

Make sure you have authenticated with the correct email ID and selected the correct project ID and region.

```
gcloud auth login
```

Enable the following APIs:

1. Artifact Registry
   1. Create a repo named `gcr.io`
2. Cloud Build

In your terminal run:

```
gcloud services enable cloudbuild.googleapis.com artifactregistry.googleapis.com
```

## Build and Deploy

Cloud Build takes the directory you are currently in (or the path you specify), looks for a Dockerfile, and builds a Docker container.

```
gcloud builds submit --tag gcr.io/[YOUR_PROJ_ID]/iris-app
```

Deploy the container to Cloud Run. You will need to enable the Cloud Run API the first time you use it.

```
gcloud run deploy iris-app --image gcr.io/[YOUR_PROJ_ID]/iris-app --platform managed --port 8501 --allow-unauthenticated
```

Once the application is deployed, update the deployed URL in your frontend source code (`streamlit_app.py`), then run:

```
streamlit run streamlit_app.py
```

---

## Modifications

The original lab classified Iris flowers. This version changes the dataset, the model, the interface, and the deployed version.

### Summary of changes

| Area | Original | Modified |
|---|---|---|
| Dataset | Iris (4 features, 3 classes) | Wine, from scikit-learn (13 features, 3 classes, 178 samples) |
| Model | Logistic Regression / Random Forest | SVM (RBF kernel) in a pipeline with `StandardScaler` |
| API response | `{"prediction": label}` | `{"prediction": label, "confidence": float}` |
| Input validation | None | Returns HTTP 400 with the missing field names |
| Frontend | 4 sliders, Iris images | 13 sliders, confidence metric and progress bar, wine-themed UI |
| Version | `latest` | `v2` image tag, deployed as a new Cloud Run revision |

### Details

- **Dataset (`src/train.py`)**: loads `load_wine()` from scikit-learn and uses an 80/20 stratified train/test split.
- **Model (`src/train.py`)**: SVMs are sensitive to feature scale, so `StandardScaler` and `SVC(kernel="rbf", probability=True)` are combined in one `Pipeline` and saved to `model/model.pkl`.
- **Prediction (`src/predict.py`)**: takes the 13 features by name and returns the predicted class and the model's confidence (the highest class probability).
- **API (`src/main.py`)**: the `/predict` route validates that all 13 fields are present and returns a 400 error with the missing field names otherwise. Flask debug mode is turned off because the service is public.
- **Frontend (`streamlit_app.py`)**: sliders are generated from a single `FEATURES` list and default to a real class 0 sample from the dataset. The result shows the predicted cultivar and the confidence.
- **Versioning**: the image is tagged `v2` and deployed as a new revision on the same Cloud Run service, so the Service URL does not change.

### Wine features

The API expects a JSON body with these 13 fields:

`alcohol`, `malic_acid`, `ash`, `alcalinity_of_ash`, `magnesium`, `total_phenols`, `flavanoids`, `nonflavanoid_phenols`, `proanthocyanins`, `color_intensity`, `hue`, `od280_od315`, `proline`

The model returns `class_0`, `class_1` or `class_2`. These are the three grape cultivars in the dataset. The frontend displays them as Cultivar 1, 2 and 3.

### Run locally

Run everything from the `FLASK_GCP_LAB` folder.

Train the model and start the API (it listens on port 8080):

```
python src/train.py
python src/main.py
```

Test the API:

```
curl -X POST http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"alcohol":14.23,"malic_acid":1.71,"ash":2.43,"alcalinity_of_ash":15.6,"magnesium":127,"total_phenols":2.8,"flavanoids":3.06,"nonflavanoid_phenols":0.28,"proanthocyanins":2.29,"color_intensity":5.64,"hue":1.04,"od280_od315":3.92,"proline":1065}'
```

Expected response: `{"confidence": ..., "prediction": "class_0"}`

Run the frontend against the local API:

```
API_URL=http://127.0.0.1:8080/predict streamlit run streamlit_app.py
```

### Deploy the new version

Train the model first so `model/model.pkl` is current, then build and deploy:

```
python src/train.py
gcloud builds submit --tag gcr.io/[YOUR_PROJ_ID]/iris-app:v2
gcloud run deploy iris-app --image gcr.io/[YOUR_PROJ_ID]/iris-app:v2 --platform managed --region [YOUR_REGION] --port 8501 --allow-unauthenticated
```

Then set `API_URL` in `streamlit_app.py` to your Service URL followed by `/predict`, and run `streamlit run streamlit_app.py`.

### Notes

- Pin the `scikit-learn` version in `requirements.txt` to the version you trained with. Loading a model pickled with a different version produces an `InconsistentVersionWarning`.
- The Dockerfile copies `model/` and `src/` into the image, so any change to the model or API code needs a rebuild and redeploy. Changes to `streamlit_app.py` do not, because the frontend runs locally.
