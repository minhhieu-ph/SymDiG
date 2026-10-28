"""Tiny synthetic CSVs with the exact NetFlow v2 schema (tests only).

python tests/synthetic.py --out tests/tmp_dataset --rows 60000
"""
import argparse
import os

import numpy as np
import pandas as pd

COLS = ["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT", "PROTOCOL", "L7_PROTO", "IN_BYTES",
        "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
        "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT", "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT",
        "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "SRC_TO_DST_SECOND_BYTES",
        "DST_TO_SRC_SECOND_BYTES", "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES",
        "RETRANSMITTED_OUT_PKTS", "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
        "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
        "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
        "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE",
        "Label", "Attack"]

SPEC = {  # dataset file -> (attack names as they appear in the CSV, relative frequency)
    "NF-ToN-IoT-v2.csv": [("Benign", .36), ("scanning", .22), ("xss", .14), ("ddos", .12), ("password", .07),
                          ("dos", .04), ("injection", .04), ("backdoor", .006), ("mitm", .003), ("ransomware", .002)],
    "NF-BoT-IoT-v2.csv": [("Benign", .05), ("DDoS", .5), ("DoS", .4), ("Reconnaissance", .045), ("Theft", .005)],
    "NF-CSE-CIC-IDS2018-v2.csv": [("Benign", .88), ("DDOS attack-HOIC", .04), ("DoS attacks-Hulk", .03),
                                  ("FTP-BruteForce", .02), ("Bot", .015), ("Infilteration", .01),
                                  ("Brute Force -Web", .005)],
    "NF-UNSW-NB15-v2.csv": [("Benign", .96), ("Exploits", .012), ("Fuzzers", .01), ("Generic", .007),
                            ("Reconnaissance", .004), ("DoS", .003), ("Analysis", .001), ("Backdoor", .001),
                            ("Shellcode", .001), ("Worms", .001)],
}


def make(path, spec, n, seed):
    rng = np.random.default_rng(seed)
    names = [s[0] for s in spec]
    p = np.array([s[1] for s in spec])
    att = rng.choice(len(names), n, p=p / p.sum())
    hosts = [f"192.168.{i // 250}.{i % 250 + 1}" for i in range(400)]
    src = rng.integers(0, 400, n)
    dst = rng.integers(0, 400, n)
    # class-specific topology: fan-in (DDoS-like) to host 0, fan-out from host 1, relay 2->3->4
    for c in range(1, len(names)):
        m = att == c
        k = c % 3
        if k == 0:
            dst[m] = c
        elif k == 1:
            src[m] = 100 + c
        else:
            half = m & (rng.random(n) < 0.5)
            src[half], dst[half] = 200 + c, 210 + c
            src[m & ~half], dst[m & ~half] = 210 + c, 220 + c
    hot = rng.random(n) < 0.3  # high-multiplicity benign bundle
    src[hot & (att == 0)], dst[hot & (att == 0)] = 5, 6
    df = pd.DataFrame({c: rng.integers(0, 2000, n) for c in COLS[4:-2]})
    for c in ("IN_BYTES", "OUT_BYTES", "SRC_TO_DST_AVG_THROUGHPUT"):
        df[c] = (rng.lognormal(5 + att * 0.4, 1.0)).astype(np.int64)
    df["TCP_FLAGS"] = rng.integers(0, 256, n)
    df.insert(0, "IPV4_SRC_ADDR", np.array(hosts)[src])
    df.insert(1, "L4_SRC_PORT", rng.integers(1, 65535, n))
    df.insert(2, "IPV4_DST_ADDR", np.array(hosts)[dst])
    df.insert(3, "L4_DST_PORT", np.where(att > 0, 80, rng.integers(1, 65535, n)))
    df["Label"] = (att > 0).astype(int)
    df["Attack"] = np.array(names)[att]
    df = df[COLS]
    df.to_csv(path, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/tmp_dataset")
    ap.add_argument("--rows", type=int, default=60000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for i, (f, spec) in enumerate(SPEC.items()):
        make(os.path.join(a.out, f), spec, a.rows, i)
        print("wrote", f)
