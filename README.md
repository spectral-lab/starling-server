# segment_spectrogram

# For Development

Create Image:
```shell
cd segment_spectrogram
docker build -t starling-server .
```

Run:
```shell
docker run -itd --name starling --mount type=bind,source="$(pwd)/src/",dst=/app/src -p 5000:5000 starling-server
```

To see logs,
```shell
docker logs -f starling
```

To see stats,
```shell
docker stats
```

To test
```shell
docker exec -it starling python3 -m unittest discover -s src
```