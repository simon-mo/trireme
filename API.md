# Websocket Objects

For the following objects, you can provide `input`, which is meant to be the raw input; or you can provide `object id`, which can be arbitrary data handle like a s3 url and having middleware to download it.

## Pred Request

```json
{
  "path": "infer",
  "input": "[optional] raw data",
  "object id": "data handle, e.g. s3 url"
}
```

## Inference Response

```json
{
  "input": "data or image url",
  "object id": "data handle, e.g. s3 url",
  "prediction": {
    "custom json object for prediction": null
  },
  "model state id": 30
}
```

## Training Request

```json
{
  "path": "train",
  "input": "[optional] raw data or image url",
  "object id": "data handle, e.g. s3 url",
  "model state id": 30,
  "label": {
    "custom json object for label": null
  }
}
```

## Cancellation Request

```json
{
  "path": "cancel",
  "input": "image url",
  "object id": "data handle, e.g. s3 url",
  "model state id": 30
}
```

## Life of a image query

```sequence
"Client" -> "Trireme": Pred Request
"Trireme"->"Client": Inference Response
"Client"->"Trireme": Training/Cancellation Request
```

# Trireme Actor Model

Trieme can "pipe" arbitrary actor together, connecting them via `asyncio.Queue`.

Training code and states can be encapsulated in the actor object that live within the aysncio pipeline. 

An actor needs to be a class that implements:
- `__init__`
- `__call__(self, input_batch)`