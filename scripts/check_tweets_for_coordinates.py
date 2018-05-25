import os
import json

def check_tweets_for_coordinates(dr):
    t_ct = 0
    c_ct = 0
    for filename in os.listdir(dr):
        if filename.endswith(".json"):
            t_ct+=1
            f = dr+filename
            data = json.load(open(f))
            coord = data["coordinates"]
            if coord != None:
                c_ct+=1
                print(coord)
    print("total tweets: %d"%t_ct)
    print("tweets with coords: %d"%c_ct)

if __name__=="__main__":
    import sys
    check_tweets_for_coordinates(sys.argv[1])

