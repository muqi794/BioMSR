import json  
  
r1 = json.load(open("result0.json"))  
r2 = json.load(open("result1.json"))  
  
best = max([r1, r2], key=lambda x: x["auc"])  
  
print("\n🏆 FINAL BEST:")  
print(best)  
  
with open("final_best.json", "w") as f:  
    json.dump(best, f, indent=4)