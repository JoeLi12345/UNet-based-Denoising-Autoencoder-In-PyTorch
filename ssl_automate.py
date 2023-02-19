#import wesad_pretraining
import wesad_finetuning
import wesad_supervised
import statistics as stats

def ssl(subject=7):
	arr = [0.50, 0.75, 0.90, 0.95, 0.98, 0.99, 0.9925, 0.995, 0.997, 0.999]
	finetuning = []
	supervised = []
	for remove_percent in arr:
		acc_arr, acc1_arr, f1_arr, f11_arr = [], [], [], []
		for j in range(5):
			print("remove percent = {}, finetuning, step {}".format(remove_percent, j))
			wesad_finetuning.init_wandb(name=f"FT_{subject}_{remove_percent}_{j}")
			tr_data = wesad_finetuning.train(subject, remove_percent=remove_percent)
			acc, acc1, f1, f11 = wesad_finetuning.test(subject, remove_percent=remove_percent)
			acc_arr.append(acc)
			acc1_arr.append(acc1)
			f1_arr.append(f1)
			f11_arr.append(f11)
			'''print("WESAD SUPERVISED BEGIN")
			wesad_supervised.init_wandb(name=f"SV_{subject}_{remove_percent}")
			wesad_supervised.train(subject, remove_percent=remove_percent, train_dataset=tr_data)
			acc, acc1, f1, f11 = wesad_supervised.test(subject, remove_percent=remove_percent)
			supervised.append([acc, acc1, f1, f11])'''
			#print(acc, acc1, f1, f11)
		finetuning.append([(stats.mean(acc_arr), stats.stdev(acc_arr)), (stats.mean(acc1_arr), stats.stdev(acc1_arr)), (stats.mean(f1_arr), stats.stdev(f1_arr)), (stats.mean(f11_arr), stats.stdev(f11_arr))])
	for i in range(len(finetuning)):
		print(arr[i], ":", finetuning[i])
	#print(supervised)

ssl()
