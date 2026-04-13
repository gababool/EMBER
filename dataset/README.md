# Dataset

Downloads the [Wildfire Dataset](https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset) from Kaggle (~10GB).

## Prerequisites

1. Create a [Kaggle account](https://www.kaggle.com)
2. Go to [kaggle.com/settings](https://kaggle.com/settings) → API → **Create New Token**
3. Set up your token:

```bash
mkdir -p ~/.config/kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_KEY"}' > ~/.config/kaggle/kaggle.json
chmod 600 ~/.config/kaggle/kaggle.json
```

## Download

```bash
./download.sh
```

Dataset will be saved to `./wildfire-dataset/`.

> **Note:** `wildfire-dataset/` is gitignored and should never be committed.