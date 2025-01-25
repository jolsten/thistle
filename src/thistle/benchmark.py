import time

from thistle.store import TLELibrary


def main():
    t0 = time.process_time()

    store = TLELibrary()
    store.load("tests/data/25544.tle")

    t1 = time.process_time()
    print(f"Elapsed Time: {t1-t0} seconds")


if __name__ == "__main__":
    main()
