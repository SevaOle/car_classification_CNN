from pathlib import Path
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split

def create_csv_file():
    FOLDER_NAME = "data"
    FOLDER_PATH = Path.cwd() / FOLDER_NAME
    DATASET_DIR = FOLDER_PATH / "car-make-model-and-generation"

    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        path = DATASET_DIR
        print("Dataset already exists:", path)
    else:
        path = kagglehub.dataset_download(
            "riotulab/car-make-model-and-generation",
            output_dir=str(DATASET_DIR),
            force_download=False,
        )
        print("Downloaded dataset to:", path)

    DATASET_DIR = DATASET_DIR / "car-dataset-200" / "riotu-cars-dataset-200"


    data = {}
    for makeDir in sorted(DATASET_DIR.iterdir()):
        # MAKE
        make = makeDir.name
        data[make] = {}
        for modelDir in sorted(makeDir.iterdir()):
            # MODEL
            model = modelDir.name
            data[make][model] = {}
            for generationDir in sorted(modelDir.iterdir()):
                # GENERATION
                generation = generationDir.name
                data[make][model][generation] = []
                print(generation)
                for image in sorted(generationDir.iterdir()):
                    data[make][model][generation].append(image.name)


    with open("data.csv", "w", encoding="utf-8") as f:
        # adds IDs as well
        f.write("filepath,make,model,generation,image,make_id,model_id,generation_id\n")
        make_id = 0
        model_id = 0
        gen_id = 0
        for make in data:
            for model in data[make]:
                for generation in data[make][model]:
                    for image in data[make][model][generation]:
                        img_path = f"{make}/{model}/{generation}/{image}"
                        f.write(f"{img_path},{make},{model},{generation},{image},{make_id},{model_id},{gen_id}\n")
                    gen_id +=1
                model_id += 1
            make_id += 1




def create_split():
    df = pd.read_csv("./data.csv")

    ### Training, validation, test split
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["generation_id"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["generation_id"]
    )

    # for safety
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    df_split = pd.concat([train_df, val_df, test_df])
    df_split.to_csv("./split.csv", index=False)

    print(df_split["split"].value_counts())
    # print(df_split.groupby("split")["generation_id"].nunique())

    # print(accuracy_score(labels_test, labels_pred))
    # print(confusion_matrix(labels_test, labels_pred))
    # print(classification_report(labels_test, labels_pred))


if __name__ == "__main__":
    create_csv_file()
    create_split()