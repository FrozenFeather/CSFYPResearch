import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

class Application:
    def __init__(self, root, inputlist, function):
        self.root = root
        self.inputlist = inputlist
        self.root.geometry('600x500')
        self.root.title("DDI query generator")
        self.function = function

        # Input labelframe
        self.labelframe = tk.LabelFrame(self.root, text="Input")
        self.labelframe.pack(side='top', fill="both", expand="yes", ipadx=10, padx=5, pady=5)

        # Original drug list Labelframe
        self.labelframe1a = tk.LabelFrame(self.labelframe, text="Original drug list")
        self.labelframe1a.pack(side='left', fill="both", expand="yes", padx=5, pady=5)

        self.scrollbar = tk.Scrollbar(self.labelframe1a)
        self.scrollbar.grid(row=0, column=1, rowspan=4, sticky=tk.E + tk.N + tk.S, padx=0)

        self.druglist = tk.Listbox(self.labelframe1a, yscrollcommand=self.scrollbar.set, height=6, width=15)
        # for line in self.inputlist:
        #     self.druglist.insert("end", str(line))
        self.druglist.grid(row=0, column=0, rowspan=4, sticky=tk.W, padx=(10, 0))
        self.scrollbar.config(command=self.druglist.yview)

        self.inputDrug = tk.StringVar()
        self.inputEntry = ttk.Combobox(self.labelframe1a, values=self.inputlist, textvariable=self.inputDrug, width=13)
        self.inputEntry.grid(row=0, column=3, padx=10, sticky=tk.S)

        tk.Button(self.labelframe1a, text="Add Drug", width=15, command=self.add_input_drug).grid(row=1, column=3, padx=10, sticky=tk.N)
        self.DeleteButton = tk.Button(self.labelframe1a, text="Delete Drug", width=15, state="disable", command=self.delete_input_drug)
        self.DeleteButton.grid(row=2, column=3, pady=10)

        self.ClearButton = tk.Button(self.labelframe1a, text="Clear Drug", width=15, command=self.clear_input_drug)
        self.ClearButton.grid(row=3, column=3)

        #Bind event in Original drug list Labelframe
        self.druglist.bind("<Button-1>", self.normal_delete_button)
        self.druglist.bind("<FocusOut>", self.disable_delete_button)
        self.ClearButton.bind("<Button-1>", self.disable_delete_button)



        # Original drug list Labelframe
        self.labelframe1b = tk.LabelFrame(self.labelframe, text="Allergy drug list")
        self.labelframe1b.pack(side='left', fill="both", expand="yes", padx=5, pady=5)

        self.scrollbar2 = tk.Scrollbar(self.labelframe1b)
        self.scrollbar2.grid(row=0, column=1, rowspan=4, sticky=tk.E + tk.N + tk.S, padx=0)

        self.allergylist = tk.Listbox(self.labelframe1b, yscrollcommand=self.scrollbar2.set, height=6, width=15)
        # for line in self.inputlist:
        #     self.allergylist.insert("end", str(line))
        self.allergylist.grid(row=0, column=0, rowspan=4, sticky=tk.W, padx=(10, 0))
        self.scrollbar2.config(command=self.allergylist.yview)

        self.allergyDrug = tk.StringVar()
        self.inputEntry2 = ttk.Combobox(self.labelframe1b, values=self.inputlist, textvariable=self.allergyDrug, width=13).grid(row=0, column=3, padx=10, sticky=tk.S)

        tk.Button(self.labelframe1b, command=self.add_allergy_drug, text="Add Drug", width=15).grid(row=1, column=3, padx=10, sticky=tk.N)

        self.DeleteButton2 = tk.Button(self.labelframe1b, text="Delete Drug", width=15, state="disable", command=self.delete_allergy_drug)
        self.DeleteButton2.grid(row=2, column=3, pady=10)


        self.ClearButton2 = tk.Button(self.labelframe1b, command=self.clear_allergy_drug, text="Clear Drug", width=15)
        self.ClearButton2.grid(row=3, column=3)

        #Bind event in Original drug list Labelframe
        self.allergylist.bind("<Button-1>", self.normal_delete_button2)
        self.allergylist.bind("<FocusOut>", self.disable_delete_button2)
        self.ClearButton2.bind("<Button-1>", self.disable_delete_button2)

        self.labelframe2 = tk.LabelFrame(root, text="Parameter")
        self.labelframe2.pack(side='top', fill="both", expand="yes", ipadx=5, padx=5, pady=5)

        self.radioValue = tk.IntVar()
        self.Radiobutton1 = tk.Radiobutton(self.labelframe2, text="Preset Minimun Distance", variable=self.radioValue,value=0)
        self.Radiobutton1.pack(side="left", padx=5, pady=2)
        self.Radiobutton1.select()
        self.Radiobutton2 = tk.Radiobutton(self.labelframe2, text="Minimun Distance:", variable=self.radioValue, value=1)
        self.Radiobutton2.pack(side="left",padx=5, pady=2)
        self.minDisEntry = tk.Entry(self.labelframe2, state="disable")
        self.minDisEntry.pack(side="left",pady=2)
        self.Radiobutton1.bind("<Button-1>", self.disable_min_dis_entry)
        self.Radiobutton2.bind("<Button-1>", self.normal_min_dis_entry)

        self.labelframe3 = tk.LabelFrame(root, text="Result")
        self.labelframe3.pack(side='top', fill="both", expand="yes", ipadx=5, padx=5, pady=5)

        self.scrollbar3 = tk.Scrollbar(self.labelframe3)
        self.scrollbar3.grid(row=0, column=1, rowspan=4, sticky=tk.E + tk.N + tk.S, padx=0)

        self.resultlist = tk.Listbox(self.labelframe3, yscrollcommand=self.scrollbar3.set, height=10, width=60)
        self.resultlist.grid(row=0, column=0, rowspan=4, sticky=tk.W, padx=(10, 0))
        self.scrollbar3.config(command=self.resultlist.yview)

        self.CalculateButton = tk.Button(self.labelframe3, text="Calculate", width=15, command=self.calculate)
        self.CalculateButton.grid(row=0, column=3, padx=10)
        self.SaveButton = tk.Button(self.labelframe3, text="Save", width=15, command=self.file_save).grid(row=1, column=3, padx=10)
        tk.Button(self.labelframe3, text="Clear", width=15, command=self.clear_result).grid(row=2, column=3, padx=10)

        self.StatusBar = tk.Label(self.root, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.StatusBar.pack(side=tk.BOTTOM, fill=tk.X)
        self.root.bind("<Button-1>", self.reset_status)

        self.root.mainloop()

    def add_input_drug(self):
        inputdrug = self.inputDrug.get()
        if inputdrug not in self.inputlist:
            self.StatusBar["text"] = "No this drug in database"
            return
        if inputdrug not in self.druglist.get(0, "end"):
            self.druglist.insert("end", inputdrug)
            self.StatusBar["text"] = "Add drug successfully"
        else:
            self.StatusBar["text"] = "This drug already exists in the list"

    def add_allergy_drug(self):
        inputdrug = self.allergyDrug.get()
        if inputdrug not in self.inputlist:
            self.StatusBar["text"] = "No this drug in database"
            return 0
        if inputdrug not in self.allergylist.get(0, "end"):
            self.allergylist.insert("end", inputdrug)
            self.StatusBar["text"] = "Add drug successfully"
        else:
            self.StatusBar["text"] = "This drug already exists in the list"

    def normal_delete_button(self, event):
        self.DeleteButton['state'] = 'normal'

    def normal_delete_button2(self, event):
        self.DeleteButton2['state'] = 'normal'

    def disable_delete_button(self, event):
        self.DeleteButton['state'] = 'disable'

    def disable_delete_button2(self, event):
        self.DeleteButton2['state'] = 'disable'

    def delete_input_drug(self):
        self.druglist.delete(self.druglist.curselection())
        self.StatusBar["text"] = "Delete drug successfully"
        self.DeleteButton['state'] = 'disable'

    def delete_allergy_drug(self):
        self.allergylist.delete(self.allergylist.curselection())
        self.StatusBar["text"] = "Delete drug successfully"
        self.DeleteButton2['state'] = 'disable'


    def clear_input_drug(self):
        self.druglist.delete(0, "end")
        self.StatusBar["text"] = "Clear drug list successfully"

    def clear_allergy_drug(self):
        self.allergylist.delete(0, "end")
        self.StatusBar["text"] = "Clear drug list successfully"

    def normal_min_dis_entry(self, event):
        self.minDisEntry['state'] = 'normal'

    def disable_min_dis_entry(self, event):
        self.minDisEntry['state'] = 'disable'

    def clear_result(self):
        self.resultlist.delete(0, "end")

    def file_save(self):
        f = tk.filedialog.asksaveasfile(mode='w', defaultextension=".txt")
        print(f)
        if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return
        for line in self.resultlist.get(0, "end"):
            f.write(str(line)+"\n")
        f.close()
        self.StatusBar["text"] = "File save"

    def reset_status(self,event):
        self.StatusBar["text"] = ""

    def calculate(self):
        #get parameter value
        if self.radioValue.get():
            minDis = float(self.minDisEntry.get())
        else:
            minDis = len(self.druglist.get(0, "end"))
        self.resultlist.insert("end", "Original Drug list: "+str(self.druglist.get(0, "end")))
        self.resultlist.insert("end", "Allergy Drug list: "+str(self.allergylist.get(0, "end")))
        self.resultlist.insert("end", "Result:")
        calculate_list = self.function(self.druglist.get(0, "end"),
                                       self.allergylist.get(0, "end"),
                                       self.inputlist,
                                       minDis)

        output = {k: v for k, v in sorted(calculate_list.items(), key=lambda item: item[1])}
        #output_formula
        for key, item in output.items():
            self.resultlist.insert("end", "Formula:  " + key + "   Distance:  " + str(item))


        # for line in calculate_list:
        #     self.resultlist.insert("end",  line)
        self.resultlist.insert("end", "")
        self.StatusBar["text"] = "Result"


