import csv
from pathlib import Path

from task1_edge_detect_rcf import main as detect_edge_rcf
import task2_skeletonization as skelet
import task4_validate_saved_edges as validate


LAMBDA = 50
THRESHOLD = 16

EDGES_SUFFIX = "_50_16"#"_edges_1"
EDGES_SUBFOLDER = "ddn/edges"# "ddn/bsds500/edges"

THIN_EDGES_SUFFIX =  "_50_16"
THIN_EDGES_SUBFOLDER = "ddn/edges_thin"
RESULTS_PATH = Path(__file__).with_name("results.csv")

def target_test():
    # THRESHS = (16, 32, 64, 128)
    # LAMS = (5, 10, 25, 50, 75)

    skelet.LAMBDA = LAMBDA
    skelet.THRESHOLD = THRESHOLD

    skelet.EDGES_SUFFIX = EDGES_SUFFIX
    skelet.EDGES_SUBFOLDER = EDGES_SUBFOLDER

    skelet.THIN_EDGES_SUFFIX = THIN_EDGES_SUFFIX
    skelet.THIN_EDGES_SUBFOLDER = THIN_EDGES_SUBFOLDER

    skelet.main()

    validate.EDGES_SUFFIX = EDGES_SUFFIX
    validate.EDGES_SUBFOLDER = EDGES_SUBFOLDER

    validate.THIN_EDGES_SUFFIX = THIN_EDGES_SUFFIX
    validate.THIN_EDGES_SUBFOLDER = THIN_EDGES_SUBFOLDER
    print("F1-SCORE")
    _, scores_f1 = validate.main(use_pq=False)
    print("PQ")
    _, scores_pq = validate.main(use_pq=True)

    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as results_file:
        writer = csv.writer(results_file)
        writer.writerows((score,) for score in scores_f1 + scores_pq)

    print(f"Результаты сохранены в {RESULTS_PATH}")

if __name__ == "__main__":
    target_test()



