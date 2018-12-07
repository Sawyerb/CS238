"""
Gernerates visualizations--uncomment sections as required
"""

import numpy as np
from matplotlib import pyplot as plt

def heatmap(data, ax, col_labels, row_labels,
			cbar_kw={}, cbarlabel="", **kwargs):
	"""
	Create a heatmap from a numpy array and two lists of labels.

	Arguments:
		data       : A 2D numpy array of shape (N,M)
		row_labels : A list or array of length N with the labels
					 for the rows
		col_labels : A list or array of length M with the labels
					 for the columns
	Optional arguments:
		ax         : A matplotlib.axes.Axes instance to which the heatmap
					 is plotted. If not provided, use current axes or
					 create a new one.
		cbar_kw    : A dictionary with arguments to
					 :meth:`matplotlib.Figure.colorbar`.
		cbarlabel  : The label for the colorbar
	All other arguments are directly passed on to the imshow call.
	"""

	if not ax:
		ax = plt.gca()

	# Plot the heatmap
	im = ax.imshow(data, **kwargs)
	
	ax.set_xticks(np.arange(data.shape[1]))
	ax.set_yticks(np.arange(data.shape[0]))
	# ... and label them with the respective list entries.
	ax.set_xticklabels(col_labels)
	ax.set_yticklabels(row_labels)

	# Create colorbar
	cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
	cbar.ax.set_ylabel(cbarlabel, rotation=-270, va="bottom", labelpad=20)

	return im, cbar

# data = np.load('basic_big_pomcpow_scores.npy')
# plt.plot(range(0, 101), data) 
# plt.xlabel('Starting Support (%)')
# plt.ylabel('Score')
# plt.savefig('basic_big_pomcpow_scores.png')


# data  = np.load('basic_big_baseline_contributions.npy')
# data = np.mean(data, 2)
# print(data)
# fig, ax = plt.subplots()

# im, cbar = heatmap(data,ax=ax, col_labels = range(1, 11), row_labels = [round(x, 2) for x in np.arange(0, 1.1, 0.1)],
#  				   cmap="Blues", cbarlabel=" Contribution")

# ax.set_title("Baseline Model")

# # axes
# plt.xlabel("Time Step")
# plt.ylabel("Initial Support (%)")
# plt.savefig('basic_big_baseline_contributions.png')
# plt.show()


# data  = np.load('basic_pftdpw_contributions.npy')
# print(data)
# fig, ax = plt.subplots()

# im, cbar = heatmap(data,ax=ax, col_labels = range(1, 11), row_labels = [round(x, 2) for x in np.arange(0, 1.1, 0.1)],
#  				   cmap="Blues", cbarlabel=" Contribution")

# ax.set_title("PFTDPW Model")

# # axes
# plt.xlabel("Time Step")
# plt.ylabel("Initial Support (%)")
# plt.savefig('basic_pftdpw_contributions.png')
# plt.show()


# models = ["baseline", "pomcpow", "pftdpw"]

# for m in models:
# 	if(m == 'pftdpw'):
# 		data = np.load('basic_' + m + '_scores.npy')
# 		plt.plot(range(0, 110, 10), data) 
# 	else:
# 		data = np.load('basic_big_' + m + '_scores.npy')
# 		data = np.mean(data, 1)
# 		plt.plot(range(0, 110, 10), data)

# plt.legend([m for m in models], loc ='lower right')
# plt.xlabel('Starting Support (%)')
# plt.ylabel('Score')
# plt.show()
# plt.savefig('basic_scores.png')	



# data  = np.load('win_reward_big_baseline_scores.npy')
# data = data.T
# print(data)
# for i in range(data.shape[0]):
# 	vals =  data[i]
# 	max_win_reward = ((100*i)+100)
# 	max_score_possible = sum(max_win_reward * ((10-n+1)/(10+1)) for n in range(11-2, -1, -1))
# 	vals /= max_score_possible
# 	data[i] = vals * 100

# data = data.T

# fig, ax = plt.subplots()

# im, cbar = heatmap(data,ax=ax, col_labels = range(100, 1100, 100), row_labels= range(0, 110, 10),
# 				   cmap="RdYlGn", cbarlabel=" % of Max Possible Score Achieved")

# ax.set_title("Baseline Model")
# #fig.tight_layout()
# # axes
# plt.xlabel("Final Win Reward")
# plt.ylabel("Initial Support (%)")
# plt.savefig("baseline_win_reward_and_init_support.png")

# data  = np.load('lose_penalty_big_pomcpow_scores.npy')
# data = data.T
# print(data)
# for i in range(data.shape[0]):
# 	vals =  data[i]
# 	max_win_reward = 500
# 	max_score_possible = sum(max_win_reward * ((10-n+1)/(10+1)) for n in range(11-2, -1, -1))
# 	vals /= max_score_possible
# 	data[i] = vals * 100

# data = data.T

# fig, ax = plt.subplots()

# im, cbar = heatmap(data,ax=ax, col_labels = range(0, 400, 100), row_labels= range(0, 110, 10),
# 				   cmap="RdYlGn", cbarlabel=" % of Max Possible Score Achieved")

# ax.set_title("POMCPOW Model")
# #fig.tight_layout()
# # axes
# plt.xlabel("Final Lose Penalty")
# plt.ylabel("Initial Support (%)")
# plt.savefig("pomcpow_lose_penalty_and_init_support.png")



# data  = np.load('transition_big_pomcpow_contributions.npy')
# data = np.mean(data, 2)
# print(data)
# fig, ax = plt.subplots()

# im, cbar = heatmap(data,ax=ax, col_labels = range(1, 101), row_labels = [round(x, 2) for x in np.arange(0.3, 0.8, 0.1)],
#  				   cmap="Blues", cbarlabel=" Contribution")

# ax.set_title("POMCPOW Model")

# # axes
# plt.xlabel("Time Step")
# plt.ylabel("Initial Support (%)")
# plt.savefig('transition_big_pomcpow_contributions.png')
# plt.show()


# models = ["baseline", "pomcpow"]

# for m in models:
# 	data = np.load('transition_big_' + m + '_scores.npy')
# 	data = np.mean(data, 1)
# 	plt.plot(range(0, 110, 10), data)

# plt.legend([m for m in models], loc ='lower right')
# plt.xlabel('Starting Support (%)')
# plt.ylabel('Score')
# plt.savefig('transition_scores.png')	


# models = ["baseline", "pomcpow"]

# for m in models:
# 	data = np.load('multi_big_' + m + '_contributions.npy')
# 	data = np.mean(data, (0, 2))
# 	plt.plot(range(0, 10), data)

# plt.legend([m for m in models], loc ='upper right')
# plt.xlabel('Time Step')
# plt.ylabel('Contribution')
# plt.show()
# #plt.savefig('multi_contributions.png')	

# models = ["baseline", "pomcpow"]

# for m in models:
# 	data = np.load('multi_big_' + m + '_scores.npy')
# 	print(data)


# models = ["baseline", "pomcpow"]

# for m in models:
# 	data = np.load('transition_big_' + m + '_scores.npy')
# 	data = np.mean(data, (1))
# 	plt.plot( [round(x, 2) for x in np.arange(0.3, 0.8, 0.1)], data)

# plt.legend([m for m in models], loc ='upper left')
# plt.xlabel('Starting Support')
# plt.ylabel('Score')
# plt.savefig('transition_scores.png')	


# data  = np.load('multi_big_pomcpow_contributions.npy')
# data = np.mean(data, 1).T
# print(data)
# fig, ax = plt.subplots()

# im, cbar = heatmap(data,ax=ax, col_labels = range(1, 11), row_labels = range(1, 11),
#  				   cmap="Blues", cbarlabel=" Contribution", vmin=0, vmax = 1600)

# ax.set_title("POMCPOW Model")

# # axes
# plt.xlabel("Time Step")
# plt.ylabel("Election")
# plt.savefig('multi_big_pomcpow_contributions.png')
# plt.show()