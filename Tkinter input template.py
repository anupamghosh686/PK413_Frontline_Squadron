#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk

fields = ('Latitude', 'Longitude', 'ATMAX', 'ATMIN', 'humidity', 
          'pressure', 'tempmax', 'tempmin', 'Output')

def outputv(entries):
    
    float(entries['Latitude'].get())
    float(entries['Longitude'].get())
    float(entries['ATMAX'].get())
    float(entries['ATMIN'].get())
    float(entries['humidity'].get())
    float(entries['pressure'].get())
    float(entries['tempmax'].get())
    float(entries['tempmin'].get())
    float(entries['Output'].get())
    
    outputval = '39.45'
    
    entries['Output'].delete(0, tk.END)
    entries['Output'].insert(0, outputval )

    print("Output: %f" % float(outputval))

class Placeholder_State(object):
     __slots__ = 'normal_color', 'normal_font', 'placeholder_text', 'placeholder_color', 'placeholder_font', 'with_placeholder'

def add_placeholder_to(entry, placeholder, color="grey", font=None):
    normal_color = entry.cget("fg")
    normal_font = entry.cget("font")
    
    if font is None:
        font = normal_font

    state = Placeholder_State()
    state.normal_color=normal_color
    state.normal_font=normal_font
    state.placeholder_color=color
    state.placeholder_font=font
    state.placeholder_text = placeholder
    state.with_placeholder=True

    def on_focusin(event, entry=entry, state=state):
        if state.with_placeholder:
            entry.delete(0, "end")
            entry.config(fg = state.normal_color, font=state.normal_font)
        
            state.with_placeholder = False

    def on_focusout(event, entry=entry, state=state):
        if entry.get() == '':
            entry.insert(0, state.placeholder_text)
            entry.config(fg = state.placeholder_color, font=state.placeholder_font)
            
            state.with_placeholder = True

    entry.insert(0, placeholder)
    entry.config(fg = color, font=font)

    entry.bind('<FocusIn>', on_focusin, add="+")
    entry.bind('<FocusOut>', on_focusout, add="+")
    
    entry.placeholder_state = state

    return state

def makeform(root, fields):
    root.title('Data Driven Agriculture')
    entries = {}
    for field in fields:
        print(field)
        row = tk.Frame(root)
        #change text and color
        lab = tk.Label(row,bg="grey", width=22, text=field+": ", anchor='w',font="-weight bold")
        ent = tk.Entry(row)
        # call function to add placeholders
        add_placeholder_to(ent, 'Enter your value...')

        #ent.insert(0, "Enter your values here")
        row.pack(side=tk.TOP, 
                 fill=tk.X, 
                 padx=5, 
                 pady=5)
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT, 
                 expand=tk.YES, 
                 fill=tk.X)

        entries[field] = ent
       
    return entries

if __name__ == '__main__':
    root = tk.Tk()
    ents = makeform(root, fields)
    b1 = tk.Button(root, text='Output',
           command=(lambda e=ents: outputv(e)))
    b1.pack(side=tk.LEFT, padx=5, pady=5)
 
    b2 = tk.Button(root, text='Quit', command=root.quit)
    b2.pack(side=tk.LEFT, padx=5, pady=5)
    root.mainloop()


# In[ ]:




