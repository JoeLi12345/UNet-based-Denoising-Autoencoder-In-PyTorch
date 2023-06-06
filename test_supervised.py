import wesad_pretraining
import wesad_finetuning
import wesad_supervised
import statistics as stats
import sys

def ssl(subject=14):
	arr = [0.0, 0.95, 0.995]
	supervised = []
	for remove_percent in arr:
		acc_arr, acc1_arr, f1_arr, f11_arr = [], [], [], []
		sv_acc_arr, sv_acc1_arr, sv_f1_arr, sv_f11_arr = [], [], [], []
		for j in range(5):
			print("remove percent = {}, supervised, step {}".format(remove_percent, j))
			wesad_supervised.init_wandb(name=f"SV_{subject}_{remove_percent}_{j}")
			wesad_supervised.train(subject, remove_percent=remove_percent)
			acc, acc1, f1, f11 = wesad_supervised.test(subject, remove_percent=remove_percent)
			sv_acc_arr.append(acc)
			sv_acc1_arr.append(acc1)
			sv_f1_arr.append(f1)
			sv_f11_arr.append(f11)
			#print(acc, acc1, f1, f11)
		#finetuning.append([(stats.mean(acc_arr), stats.stdev(acc_arr)), (stats.mean(acc1_arr), stats.stdev(acc1_arr)), (stats.mean(f1_arr), stats.stdev(f1_arr)), (stats.mean(f11_arr), stats.stdev(f11_arr))])
		supervised.append([(stats.mean(sv_acc_arr), stats.stdev(sv_acc_arr)), (stats.mean(sv_acc1_arr), stats.stdev(sv_acc1_arr)), (stats.mean(sv_f1_arr), stats.stdev(sv_f1_arr)), (stats.mean(sv_f11_arr), stats.stdev(sv_f11_arr))])
	'''print("FINETUNING")
	for i in range(len(finetuning)):
		print(arr[i], ":", finetuning[i])'''
	print("SUPERVISED")
	for i in range(len(supervised)):
		print(arr[i], ":", supervised[i])

ssl()
