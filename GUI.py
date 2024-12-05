from random import randint

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import networkx as nx
import customtkinter

# from Main import *

from MDPSolver import MDPSolver
from Reward import Reward
from State import State
from Song import Song
from DataFetcher import *

mini_pad = 3
pos = {}


def draw_graph(all_states, selected = None):
    graph = nx.MultiDiGraph()

    for s in all_states:
        graph.add_node(s.name)

    for s in all_states:
        probabilities = s.transition_probabilities
        for neighbor, i in zip(s.neighbours, range(len(s.neighbours))):
            graph.add_edge(s.name, neighbor.name, weight=round(probabilities[i], 3))

    global pos
    if not pos:
        pos = nx.spring_layout(graph)

    node_colors = ['lightblue' if s.name == selected else '#2185C4' for s in all_states]
    plt.figure(figsize=(5, 5))
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, node_size=5000, arrows=True, font_weight='bold')

    labels = {s.name: f"\n\nR: {round(s.reward, 3)}" for s in all_states}
    nx.draw_networkx_labels(graph, pos, labels=labels)

    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    return plt.gcf()


def get_reward_simulation():
    selected_state = state_dropdown.get()

    slider_value = round(slider.get(), 2)

    repeat = repeat_check.get()
    share = share_check.get()
    add = add_check.get()
    remove = remove_check.get()

    good_actions_map = {
        'repeat': False,
        'added_song': False,
        'shared': False,
    }
    bad_actions_map = {
        'skip': False,
        'removed_song': False
    }

    slider_value /= 100.0

    if repeat == 1:
        good_actions_map['repeat'] = True
    if share == 1:
        good_actions_map['shared'] = True
    if add == 1:
        good_actions_map['added_song'] = True
    if remove == 1:
        bad_actions_map['removed_song'] = True
    if slider_value < 0.05:
        bad_actions_map['skip'] = True

    positive = 0
    negative = 0
    for key in good_actions_map.keys():
        if good_actions_map[key]:
            positive += 1

    for key in bad_actions_map.keys():
        if bad_actions_map[key]:
            negative -= 1

    if negative == -2:
        reward_sim = (positive + negative + slider_value) / 2
    else:
        reward_sim = (positive + negative + slider_value) / 4

    states[selected_state].set_reward(reward_sim)
    return selected_state


def update_probabilities():
    selected = get_reward_simulation()

    for state_name, entries in alpha_entries.items():
        alpha_values = [float(entry.get()) for entry in entries]
        states[state_name].set_alpha_values(alpha_values)

        new_probabilities = states[state_name].transition_probabilities
        prob_label = prob_labels[state_name]
        prob_text = "Probabilities:\t\t" + "\t\t".join(str(round(prob, 3)) for prob in new_probabilities)
        prob_label.configure(text=prob_text)

    fig = draw_graph(states.values(), selected)
    display_graph(fig)

    solver = MDPSolver(states)
    solutions = solver.get_optimal_policy()


    for widget in action_frame.winfo_children():
        widget.destroy()

    action_frame_label = customtkinter.CTkLabel(master=action_frame, text="Optimal policies", bg_color="#1F69A3",
                                                font=('Roboto', 20))
    action_frame_label.pack(pady=10, expand=True, fill="both")

    for state, best_action in solutions.items():
        best_action_frame = customtkinter.CTkFrame(master=action_frame)
        best_action_frame.pack(padx=5, pady=5, fill="both", expand=True)
        txt = "Move from " + str(state) + " to " + str(best_action)
        best_action_label = customtkinter.CTkLabel(master=best_action_frame, text=txt, font=('Roboto', 16))
        best_action_label.pack(pady=5)


def display_graph(fig):
    for widget in graph_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)


def display_slider_value(value):
    rounded_value = round(value, 2)
    slider_value_label.configure(text=str(rounded_value))


def on_state_selected(selected_value):
    print(f"Selected state: {selected_value}")


if __name__ == "__main__":
    color2 = '#FFDC7F'
    color1 = '#78B7D0'
    # print("Enter the number of the states")
    state_num = 4
    state_names = []
    for i in range(state_num):
        state_names.append("State " + str(i))

    all_songs = []
    for entry in dataset:
        song = Song(attributes, entry)
        all_songs.append(song)

    rewards = []
    last_songs = []
    for i in range(len(state_names)):
        prev_songs = [all_songs[randint(0, len(all_songs) - 1)], all_songs[randint(0, len(all_songs) - 1)]]
        last_songs.append(prev_songs)
        reward = Reward(all_songs[randint(0, len(all_songs) - 1)], prev_songs)
        rewards.append(reward)

    states = {}
    for i in range(len(state_names)):
        state = State(state_names[i], rewards[i], last_songs[i], [])
        states[state_names[i]] = state

    for i in range(state_num):
        current_state = f"State {i}"
        neighbours = []
        take_care = 0
        while len(neighbours) < state_num / 2 and take_care < len(state_names):
            generated = take_care
            take_care += 1
            name = f"State {generated}"
            if name != current_state and name not in neighbours and states[current_state] not in states[
                name].neighbours:
                neighbours.append(name)
                states[current_state].add_neighbour(states[name])

    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme('dark-blue')
    root = customtkinter.CTk()
    root.geometry("800x600")

    left_frame = customtkinter.CTkFrame(master=root)
    left_frame.pack(padx=10, pady=10, expand=True, fill="both", side="left")

    graph_frame = customtkinter.CTkFrame(master=left_frame)
    graph_frame.pack(padx=10, pady=10, fill="both", expand=True, side="top")

    # show actions info
    stats_frame = customtkinter.CTkFrame(master=left_frame)
    stats_frame.pack(padx=10, pady=10, fill="both", expand=True, side="bottom")

    reward_generating_frame = customtkinter.CTkFrame(master=stats_frame)
    reward_generating_frame.pack(padx=10, pady=10, fill="both", side="left", expand=True)

    # dropdown menu to select a state
    state_dropdown = customtkinter.CTkOptionMenu(
        master=reward_generating_frame,
        values=state_names
    )
    state_dropdown.pack(padx=20, pady=20)

    # listening percentage slider and label
    slider = customtkinter.CTkSlider(master=reward_generating_frame, from_=0, to=100, command=display_slider_value)
    slider.pack(padx=20, pady=20)
    slider_value_label = customtkinter.CTkLabel(master=reward_generating_frame, text="Listening percentage")
    slider_value_label.pack(pady=10)

    checkboxes_reward_frame = customtkinter.CTkFrame(master=reward_generating_frame)
    checkboxes_reward_frame.pack(padx=10, pady=10)

    repeat_check = customtkinter.CTkCheckBox(master=checkboxes_reward_frame, text="Repeat")
    repeat_check.pack(padx=5, pady=5, side='left', anchor="center")

    share_check = customtkinter.CTkCheckBox(master=checkboxes_reward_frame, text="Share")
    share_check.pack(padx=5, pady=5, side='left', anchor="center")

    add_check = customtkinter.CTkCheckBox(master=checkboxes_reward_frame, text="Add")
    add_check.pack(padx=5, pady=5, side='left', anchor="center")

    remove_check = customtkinter.CTkCheckBox(master=checkboxes_reward_frame, text="Remove")
    remove_check.pack(padx=5, pady=5, anchor="center")

    generate_reward_btn = customtkinter.CTkButton(master=reward_generating_frame, text="Generate reward",
                                                  command=update_probabilities)
    generate_reward_btn.pack(padx=10, pady=10)

    # TODO: show best action for each state
    action_frame = customtkinter.CTkFrame(master=stats_frame)
    action_frame.pack(padx=10, pady=10, fill="both", expand=True, side="left")

    solver = MDPSolver(states)
    print(solver.get_optimal_policy())

    for i in range(len(state_names)):
        states[state_names[i]].random_rec()

    action_frame_label = customtkinter.CTkLabel(master=action_frame, text="Optimal policies", bg_color=color1, font=('Roboto', 20))
    action_frame_label.pack(pady=10, expand=True, fill="both")

    solutions = solver.get_optimal_policy()
    actions = ""
    for state, best_action in zip(solutions.keys(), solutions.values()):
        best_action_frame = customtkinter.CTkFrame(master=action_frame)
        best_action_frame.pack(padx=5, pady=5, fill="both", expand=True)
        txt = "Move from " + str(state) + " to " + str(best_action)
        best_action_label = customtkinter.CTkLabel(master=best_action_frame, text=txt, font=('Roboto', 16))
        best_action_label.pack(pady=5)

    policies = customtkinter.CTkFrame(master=action_frame)
    policies.pack(padx=5, pady=5)

    policies_label = customtkinter.CTkLabel(master=policies, text=actions)
    policies_label.pack(pady=10)


    control_frame = customtkinter.CTkFrame(master=root)
    control_frame.pack(side="right", fill="y")

    alpha_entries = {}
    prob_labels = {}

    for i in range(state_num):
        row = customtkinter.CTkFrame(master=control_frame)
        row.pack(padx=10, pady=10,  expand=True)
        name = f"State {i}"

        label = customtkinter.CTkLabel(master=row, text=name)
        label.pack(padx=mini_pad, pady=mini_pad, side="top")

        state = states[name]
        state_information = customtkinter.CTkFrame(master=row)
        state_information.pack(padx=mini_pad, pady=mini_pad)

        # Neighbours information
        neighbour_states = customtkinter.CTkFrame(master=state_information)
        neighbour_states.pack(padx=mini_pad, pady=mini_pad)
        neighbours = state.get_neighbours()

        neighbours_text = "Neighbours:\t\t" + "\t\t".join(neighbour.name for neighbour in neighbours)
        label_neigh = customtkinter.CTkLabel(master=neighbour_states, text=neighbours_text, text_color=color1)
        label_neigh.pack(padx=mini_pad, pady=mini_pad)

        alpha_frame = customtkinter.CTkFrame(master=state_information)
        alpha_frame.pack(padx=mini_pad, pady=mini_pad, expand=True)

        alpha_label = customtkinter.CTkLabel(master=alpha_frame, text="Alpha values:  ", text_color=color1)
        alpha_label.pack(padx=mini_pad, pady=mini_pad, side="left")

        current_alpha_entries = []
        for alpha in state.alpha:
            alpha_input = customtkinter.CTkEntry(master=alpha_frame, width=50, placeholder_text=str(alpha), text_color=color2)
            alpha_input.pack(padx=35, pady=mini_pad, side="left")
            current_alpha_entries.append(alpha_input)

        alpha_entries[name] = current_alpha_entries

        prob_frame = customtkinter.CTkFrame(master=state_information)
        prob_frame.pack(padx=mini_pad, pady=mini_pad)

        prob_text = "Probabilities:\t\t" + "\t\t".join(str(round(prob, 3)) for prob in state.transition_probabilities)
        prob_label = customtkinter.CTkLabel(master=prob_frame, text=prob_text, text_color=color2)
        prob_label.pack(padx=mini_pad, pady=mini_pad)

        prob_labels[name] = prob_label

    generate = customtkinter.CTkButton(master=control_frame, text="Update Probabilities", command=update_probabilities)
    generate.pack(padx=mini_pad, pady=mini_pad)

    initial_fig = draw_graph(states.values())
    display_graph(initial_fig)

    root.mainloop()

