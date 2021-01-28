if __name__ == "__main__":
    pass

def find_outdir():
    import os
    # get the correct directories
    dirs = ["/Users/martijac/Documents/Frailty/frailty_classifier/output/",
            "/media/drv2/andrewcd2/frailty/output/", "/share/gwlab/frailty/output/"]
    for d in dirs:
        if os.path.exists(d):
            outdir = d
    return(outdir)