# Deploy (Render, Docker)

`render.yaml` declares this as a Docker web service using the existing
`Dockerfile` — no extra setup needed beyond setting `MONGODB_URI`.

## Steps

1. Push this repo to GitHub.
2. On Render: **New > Blueprint**, point it at this repo.
3. Set `MONGODB_URI` to the same Atlas connection string TLCN-BE uses
   (read access to the `travela` database is enough).
4. Deploy. Check `https://<your-service>.onrender.com/health` returns OK.
5. Copy this service's URL into TLCN-BE's `RECOMMENDATION_API` env var.

## Important: do not use the free plan

`requirements.txt` includes `tensorflow==2.15.0` for the DeepFM model. Render's
free web service tier caps at 512MB RAM, which TensorFlow alone tends to
exceed on load — the service will likely crash-loop (OOM) on the free tier.
`render.yaml` is set to `plan: starter` (paid) for this reason. If you want to
try the free tier anyway to confirm, expect it to fail and not be a bug in
the code.
