import os
from pathlib import Path
import queue
import shutil
import threading
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from .analyze import analyze_file

BASE=Path(__file__).resolve().parents[1]

class App:
    def __init__(self,root):
        self.root=root; self.bus=queue.Queue(); self.stop=threading.Event(); self.running=False; self.result=None
        root.title('Проверка адресов ГАР'); root.geometry('780x420'); root.minsize(680,400)
        body=ttk.Frame(root,padding=24); body.pack(fill='both',expand=True)
        ttk.Label(body,text='Проверка адресов из Excel',font=('Segoe UI',18,'bold')).pack(anchor='w')
        ttk.Label(body,text='Локальная модель и индекс ГАР · административная иерархия').pack(anchor='w',pady=(4,20))
        self.input=tk.StringVar(); self.output=tk.StringVar(); self.status=tk.StringVar(value='Выберите заполненный шаблон Excel.')
        for label,var,command in [('Входной Excel',self.input,self.choose_input),('Сохранить результат',self.output,self.choose_output)]:
            ttk.Label(body,text=label).pack(anchor='w')
            frame=ttk.Frame(body); frame.pack(fill='x',pady=(4,12))
            ttk.Entry(frame,textvariable=var).pack(side='left',fill='x',expand=True)
            ttk.Button(frame,text='Обзор…',command=command).pack(side='right',padx=(8,0))
        buttons=ttk.Frame(body); buttons.pack(fill='x',pady=10)
        self.start=ttk.Button(buttons,text='Проверить адреса',command=self.run); self.start.pack(side='left')
        self.cancel=ttk.Button(buttons,text='Отмена',command=self.stop.set,state='disabled'); self.cancel.pack(side='left',padx=8)
        ttk.Button(buttons,text='Сохранить шаблон',command=self.template).pack(side='left')
        self.open=ttk.Button(buttons,text='Открыть результат',command=self.open_result,state='disabled'); self.open.pack(side='right')
        self.bar=ttk.Progressbar(body,mode='determinate'); self.bar.pack(fill='x',pady=12)
        ttk.Label(body,textvariable=self.status,wraplength=710).pack(anchor='w')
        root.protocol('WM_DELETE_WINDOW',self.close); root.after(100,self.poll)
    def choose_input(self):
        if self.running:return
        path=filedialog.askopenfilename(filetypes=[('Excel','*.xlsx')])
        if path:
            self.input.set(path); self.output.set(str(Path(path).with_name(Path(path).stem+'_результат.xlsx')))
    def choose_output(self):
        if self.running:return
        path=filedialog.asksaveasfilename(defaultextension='.xlsx',filetypes=[('Excel','*.xlsx')])
        if path:self.output.set(path)
    def template(self):
        path=filedialog.asksaveasfilename(initialfile='Шаблон_адресов.xlsx',defaultextension='.xlsx')
        if path:
            try:shutil.copyfile(BASE/'Шаблон_адресов.xlsx',path)
            except Exception as e:messagebox.showerror('Ошибка',str(e))
    def run(self):
        if self.running:return
        source,target=self.input.get(),self.output.get()
        if not Path(source).is_file() or not target:
            messagebox.showerror('Выбор файла','Укажите входной Excel и путь результата.');return
        if Path(source).resolve()==Path(target).resolve():
            messagebox.showerror('Выбор файла','Нельзя перезаписывать исходный файл.');return
        if Path(target).exists() and not messagebox.askyesno('Результат уже существует','Заменить существующий файл результата?'):return
        self.running=True;self.result=None;self.stop.clear();self.start.configure(state='disabled');self.cancel.configure(state='normal');self.open.configure(state='disabled')
        self.status.set('Подготовка…');self.bar.configure(value=0)
        def worker():
            try:
                summary=analyze_file(source,target,BASE/'data/gar.gari',BASE/'data/best.npz',lambda d,t,s:self.bus.put(('progress',d,t,s)),self.stop.is_set)
                self.bus.put(('done',target,summary))
            except InterruptedError:self.bus.put(('cancel',))
            except Exception as e:self.bus.put(('error',str(e)))
        threading.Thread(target=worker,daemon=True).start()
    def poll(self):
        try:
            while True:
                event=self.bus.get_nowait()
                if event[0]=='progress':self.bar.configure(maximum=event[2],value=event[1]);self.status.set(event[3]);continue
                self.running=False;self.start.configure(state='normal');self.cancel.configure(state='disabled')
                if event[0]=='done':
                    self.result=event[1];s=event[2];self.open.configure(state='normal');self.status.set(f'Готово. Всего: {s["total"]}. Валидных: {s["valid"]}. Невалидных: {s["invalid"]}.')
                elif event[0]=='cancel':self.status.set('Проверка отменена. Результат не сохранён.')
                else:self.status.set('Ошибка обработки.');messagebox.showerror('Ошибка',event[1])
        except queue.Empty:pass
        self.root.after(100,self.poll)
    def open_result(self):
        try:os.startfile(self.result)
        except OSError as e:messagebox.showerror('Не удалось открыть файл',str(e))
    def close(self):
        if self.running:
            messagebox.showinfo('Выполняется проверка','Нажмите «Отмена» и дождитесь завершения текущего адреса.');return
        self.root.destroy()

def main():
    root=tk.Tk();App(root);root.mainloop()
if __name__=='__main__':main()
