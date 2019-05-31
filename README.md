# starling-server

## For Development

Create Image:
```shell
cd starling-server
docker build -t starling-server .
```

Run:
```shell
chmod +x run.sh && ./run.sh
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
docker exec -it starling python3 -m unittest discover -v -s src
```
