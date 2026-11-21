#!/usr/bin/env python3
"""Minimal client for axdiffusion-server (standard library only).

    python3 client.py --url http://BOARD_IP:18500 --prompt "a lighthouse at sunset" --out out.png
    python3 client.py --url http://BOARD_IP:18500 --prompt "..." --lora my_style:0.8 --api a1111
"""
import argparse
import base64
import json
import urllib.request


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18500")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--lora", action="append", default=[], help="NAME[:SCALE], repeatable")
    ap.add_argument("--api", choices=["openai", "a1111"], default="openai")
    ap.add_argument("--out", default="out.png")
    a = ap.parse_args()

    loras = []
    for item in a.lora:
        name, _, scale = item.partition(":")
        loras.append({"name": name, "scale": float(scale or 1.0)})

    if a.api == "openai":
        body = {"prompt": a.prompt, "size": f"{a.size}x{a.size}", "loras": loras}
        if a.seed >= 0:
            body["seed"] = a.seed
        d = post(a.url + "/v1/images/generations", body)
        png = base64.b64decode(d["data"][0]["b64_json"])
        print(json.dumps(d["info"][0]))
    else:
        prompt = a.prompt + "".join(f" <lora:{l['name']}:{l['scale']}>" for l in loras)
        d = post(a.url + "/sdapi/v1/txt2img", {"prompt": prompt, "width": a.size, "height": a.size, "seed": a.seed})
        png = base64.b64decode(d["images"][0])
        print(d["info"])
    with open(a.out, "wb") as f:
        f.write(png)
    print("saved", a.out)


if __name__ == "__main__":
    main()
