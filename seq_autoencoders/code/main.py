import cluster
import seqtoseq
import test
def main():
    seqtoseq.train()
    test.test()

    cluster.cluster()


if __name__ == "__main__":
        main()