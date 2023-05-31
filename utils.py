import time


# tic and toc functions for measuring time
def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc(pName=''):
    if 'startTime_for_tictoc' in globals():
        delta = time.time() - startTime_for_tictoc
        print("{}: Elapsed time is ".format(pName) + str(int(delta)) + " seconds, or " + str(
            round(delta / 60, 2)) + " minutes")
    else:
        delta = -1
        print("Toc: start time not set")
    return delta
