import time

in_file = "/media/tcastillo/SSD2/fr_ex.txt/fr.txt"
out_file = "/media/tcastillo/SSD2/fr_ex.txt/fr_clean.txt"
encoding = "utf-8"

in_con = open(in_file, "r", encoding=encoding, errors="ignore")
out_con = open(out_file, "w+", encoding=encoding)

AUTHORIZED_UNICODE = set(
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    '0123456789'
    ' !"#$%&\'`()*+,-./:;<=>?@[\\]^_{|}~'
    'ÀàÂâÄäÇçÉéÈèÊêËëÎîÏïÔôÖöÙùÛûÜüÆæŒœ'
    '€£¥•·²³≠±×÷√π'

)

if __name__ == "__main__":
    t1 = time.time()
    for i, line in enumerate(in_con):
        if "***" in line:
            out_con.write(" \n \n ")
        filtered_line = ''.join([char for char in line.strip() if char in AUTHORIZED_UNICODE])
        out_con.write(filtered_line)
        if i % 1000 == 0:
            print(f"line: {i}")
    t2 = time.time()
    print(str(t2 - t1) + "s")
    in_con.close()
    out_con.close()
