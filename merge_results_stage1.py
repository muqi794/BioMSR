import json  
  
# 两个人的结果  
file1 = "tuning_results1.json"  
file2 = "tuning_results2.json"  
  
with open(file1, "r") as f:  
    r1 = json.load(f)  
  
with open(file2, "r") as f:  
    r2 = json.load(f)  
  
# 合并  
all_results = r1 + r2  
  
# 按 AUC 排序  
all_results_sorted = sorted(all_results, key=lambda x: x["auc"], reverse=True)  
  
# ⭐ Top 2  
top2 = all_results_sorted[:2]  
  
print("\n🏆 GLOBAL TOP 2:")  
for t in top2:  
    print(t)  
  
# 保存  
with open("all_results.json", "w") as f:  
    json.dump(all_results, f, indent=4)  
  
with open("top2_global.json", "w") as f:  
    json.dump(top2, f, indent=4)