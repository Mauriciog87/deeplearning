import customtkinter as ctk
from typing import Dict, List, Tuple, Optional, Callable, Any


class SessionPanel(ctk.CTkFrame):
    
    def __init__(
        self, 
        master,
        sessions: List[Dict],
        on_session_change: Optional[Callable] = None,
        on_new_session: Optional[Callable] = None,
        on_load_session: Optional[Callable] = None,
        on_delete_session: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(master, **kwargs)
        
        self.on_session_change = on_session_change
        self.on_new_session = on_new_session
        self.on_load_session = on_load_session
        self.on_delete_session = on_delete_session
        
        self.grid_columnconfigure(0, weight=1)
        
        self.title_label = ctk.CTkLabel(
            self, 
            text="Session", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=4, padx=10, pady=(10, 5), sticky="w")
        
        session_names = [s.get("name", f"Session {s.get('id', '?')}") for s in sessions]
        self.session_var = ctk.StringVar(value=session_names[0] if session_names else "No sessions")
        
        self.session_dropdown = ctk.CTkOptionMenu(
            self,
            values=session_names if session_names else ["No sessions"],
            variable=self.session_var,
            command=self._on_session_selected,
            width=200
        )
        self.session_dropdown.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        self.new_btn = ctk.CTkButton(
            self,
            text="New",
            width=60,
            command=self._on_new_clicked
        )
        self.new_btn.grid(row=1, column=1, padx=5, pady=5)
        
        self.load_btn = ctk.CTkButton(
            self,
            text="Load",
            width=60,
            command=self._on_load_clicked
        )
        self.load_btn.grid(row=1, column=2, padx=2, pady=5)
        
        self.delete_btn = ctk.CTkButton(
            self,
            text="Delete",
            width=60,
            fg_color="#8B0000",
            hover_color="#A52A2A",
            command=self._on_delete_clicked
        )
        self.delete_btn.grid(row=1, column=3, padx=(2, 10), pady=5)
        
        self.info_label = ctk.CTkLabel(
            self,
            text="0 spins",
            font=ctk.CTkFont(size=12)
        )
        self.info_label.grid(row=2, column=0, columnspan=4, padx=10, pady=(0, 10), sticky="w")
    
    def _on_session_selected(self, choice: str):
        if self.on_session_change:
            self.on_session_change(choice)
    
    def _on_new_clicked(self):
        if self.on_new_session:
            self.on_new_session()
    
    def _on_load_clicked(self):
        if self.on_load_session:
            self.on_load_session()
    
    def _on_delete_clicked(self):
        if self.on_delete_session:
            self.on_delete_session()
    
    def update_sessions(self, sessions: List[Dict]):
        session_names = [s.get("name", f"Session {s.get('id', '?')}") for s in sessions]
        self.session_dropdown.configure(values=session_names if session_names else ["No sessions"])
        if session_names:
            self.session_var.set(session_names[0])
    
    def set_spin_count(self, count: int):
        self.info_label.configure(text=f"{count} spins")


class NumberInputPanel(ctk.CTkFrame):
    
    def __init__(
        self, 
        master,
        on_add_number: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__(master, **kwargs)
        
        self.on_add_number = on_add_number
        
        self.grid_columnconfigure(0, weight=1)
        
        self.title_label = ctk.CTkLabel(
            self, 
            text="Add Number", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")
        
        self.number_entry = ctk.CTkEntry(
            self,
            placeholder_text="0-36",
            width=100
        )
        self.number_entry.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.number_entry.bind("<Return>", self._on_enter_pressed)
        
        self.add_btn = ctk.CTkButton(
            self,
            text="Add",
            width=70,
            command=self._on_add_clicked
        )
        self.add_btn.grid(row=1, column=1, padx=(5, 10), pady=10)
        
        self.last_numbers_label = ctk.CTkLabel(
            self,
            text="Last: -",
            font=ctk.CTkFont(size=12)
        )
        self.last_numbers_label.grid(row=2, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="w")
    
    def _on_enter_pressed(self, event):
        self._on_add_clicked()
    
    def _on_add_clicked(self):
        try:
            number = int(self.number_entry.get())
            if 0 <= number <= 36:
                if self.on_add_number:
                    self.on_add_number(number)
                self.number_entry.delete(0, "end")
        except ValueError:
            pass
    
    def update_last_numbers(self, numbers: List[int]):
        if numbers:
            display = ", ".join(str(n) for n in numbers[-5:])
            self.last_numbers_label.configure(text=f"Last: {display}")


class PredictionDisplay(ctk.CTkFrame):
    
    RED_NUMBERS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        self.title_label = ctk.CTkLabel(
            self, 
            text="Prediction", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.title_label.grid(row=0, column=0, padx=10, pady=(10, 5))
        
        self.number_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.number_frame.grid(row=1, column=0, padx=10, pady=10)
        
        self.number_label = ctk.CTkLabel(
            self.number_frame,
            text="--",
            font=ctk.CTkFont(size=72, weight="bold"),
            text_color="gray"
        )
        self.number_label.pack()
        
        self.confidence_label = ctk.CTkLabel(
            self.number_frame,
            text="Confidence: --",
            font=ctk.CTkFont(size=14)
        )
        self.confidence_label.pack()
        
        self.predictor_label = ctk.CTkLabel(
            self.number_frame,
            text="Source: --",
            font=ctk.CTkFont(size=12)
        )
        self.predictor_label.pack()
        
        self.top3_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.top3_frame.grid(row=2, column=0, padx=10, pady=(5, 10))
        
        self.top3_labels = []
        for i in range(3):
            lbl = ctk.CTkLabel(
                self.top3_frame,
                text="--",
                font=ctk.CTkFont(size=24),
                width=50
            )
            lbl.grid(row=0, column=i, padx=5)
            self.top3_labels.append(lbl)
    
    def _get_color_for_number(self, number: int) -> str:
        if number == 0:
            return "#228B22"
        elif number in self.RED_NUMBERS:
            return "#DC143C"
        else:
            return "#FFFFFF"
    
    def update_prediction(
        self, 
        number: int, 
        confidence: float, 
        predictor: str,
        top_numbers: List[Tuple[int, float]]
    ):
        color = self._get_color_for_number(number)
        self.number_label.configure(text=str(number), text_color=color)
        self.confidence_label.configure(text=f"Confidence: {confidence:.1%}")
        self.predictor_label.configure(text=f"Source: {predictor}")
        
        for i, lbl in enumerate(self.top3_labels):
            if i < len(top_numbers):
                num, prob = top_numbers[i]
                num_color = self._get_color_for_number(num)
                lbl.configure(text=str(num), text_color=num_color)
            else:
                lbl.configure(text="--", text_color="gray")
    
    def clear(self):
        self.number_label.configure(text="--", text_color="gray")
        self.confidence_label.configure(text="Confidence: --")
        self.predictor_label.configure(text="Source: --")
        for lbl in self.top3_labels:
            lbl.configure(text="--", text_color="gray")


class ProbabilityBars(ctk.CTkFrame):
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.grid_columnconfigure(1, weight=1)
        
        self.title_label = ctk.CTkLabel(
            self, 
            text="Category Probabilities", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 5), sticky="w")
        
        self.categories = [
            ("Color", "color", {"red": "#DC143C", "black": "#2F2F2F", "green": "#228B22"}),
            ("Parity", "parity", {"odd": "#4169E1", "even": "#FF8C00"}),
            ("High/Low", "high_low", {"high": "#9932CC", "low": "#20B2AA"}),
            ("Dozen", "dozen", {1: "#FF6347", 2: "#4682B4", 3: "#32CD32"}),
            ("Column", "column", {1: "#FF69B4", 2: "#00CED1", 3: "#FFD700"})
        ]
        
        self.bars: Dict[str, Dict] = {}
        
        for i, (name, key, colors) in enumerate(self.categories):
            row = i + 1
            
            label = ctk.CTkLabel(self, text=name, width=70, anchor="w")
            label.grid(row=row, column=0, padx=(10, 5), pady=3, sticky="w")
            
            bar_frame = ctk.CTkFrame(self, fg_color="gray20", height=25)
            bar_frame.grid(row=row, column=1, padx=5, pady=3, sticky="ew")
            bar_frame.grid_propagate(False)
            
            value_label = ctk.CTkLabel(self, text="--", width=80)
            value_label.grid(row=row, column=2, padx=(5, 10), pady=3)
            
            self.bars[key] = {
                "frame": bar_frame,
                "value_label": value_label,
                "colors": colors,
                "segments": {}
            }
    
    def update_category(self, category: str, value: Any, probability: float, all_probs: Dict[Any, float]):
        if category not in self.bars:
            return
        
        bar_info = self.bars[category]
        bar_frame = bar_info["frame"]
        colors = bar_info["colors"]
        
        for widget in bar_frame.winfo_children():
            widget.destroy()
        
        bar_frame.grid_columnconfigure(list(range(len(all_probs))), weight=0)
        
        total_width = bar_frame.winfo_width() or 200
        
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        col = 0
        for val, prob in sorted_probs:
            if prob <= 0:
                continue
            
            width = max(int(total_width * prob), 2)
            color = colors.get(val, "gray50")
            
            segment = ctk.CTkFrame(
                bar_frame,
                fg_color=color,
                width=width,
                height=23
            )
            segment.grid(row=0, column=col, sticky="ns")
            segment.grid_propagate(False)
            col += 1
        
        bar_info["value_label"].configure(text=f"{value}: {probability:.1%}")
    
    def clear(self):
        for key, bar_info in self.bars.items():
            for widget in bar_info["frame"].winfo_children():
                widget.destroy()
            bar_info["value_label"].configure(text="--")


class TopNumbersFan(ctk.CTkFrame):
    
    RED_NUMBERS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
    
    def __init__(self, master, count: int = 10, **kwargs):
        super().__init__(master, **kwargs)
        
        self.count = count
        
        self.title_label = ctk.CTkLabel(
            self, 
            text=f"Top {count} Numbers", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.title_label.pack(padx=10, pady=(10, 5))
        
        self.numbers_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.numbers_frame.pack(padx=10, pady=10, fill="x")
        
        for i in range(count):
            self.numbers_frame.grid_columnconfigure(i, weight=1)
        
        self.number_widgets: List[Dict] = []
        
        for i in range(count):
            cell_frame = ctk.CTkFrame(
                self.numbers_frame,
                fg_color="gray30",
                corner_radius=8
            )
            cell_frame.grid(row=0, column=i, padx=2, pady=2, sticky="nsew")
            
            num_label = ctk.CTkLabel(
                cell_frame,
                text="--",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color="white"
            )
            num_label.pack(pady=(5, 0))
            
            prob_label = ctk.CTkLabel(
                cell_frame,
                text="",
                font=ctk.CTkFont(size=9),
                text_color="gray"
            )
            prob_label.pack(pady=(0, 5))
            
            self.number_widgets.append({
                "frame": cell_frame,
                "number": num_label,
                "prob": prob_label
            })
    
    def _get_bg_color_for_number(self, number: int) -> str:
        if number == 0:
            return "#228B22"
        elif number in self.RED_NUMBERS:
            return "#DC143C"
        else:
            return "#2F2F2F"
    
    def update_numbers(self, top_numbers: List[Tuple[int, float]]):
        for i, widget in enumerate(self.number_widgets):
            if i < len(top_numbers):
                num, prob = top_numbers[i]
                bg_color = self._get_bg_color_for_number(num)
                widget["frame"].configure(fg_color=bg_color)
                widget["number"].configure(text=str(num))
                widget["prob"].configure(text=f"{prob:.1%}")
            else:
                widget["frame"].configure(fg_color="gray30")
                widget["number"].configure(text="--")
                widget["prob"].configure(text="")
    
    def clear(self):
        for widget in self.number_widgets:
            widget["frame"].configure(fg_color="gray30")
            widget["number"].configure(text="--")
            widget["prob"].configure(text="")


class PredictorStatsTable(ctk.CTkScrollableFrame):
    
    def __init__(self, master, **kwargs):
        super().__init__(master, orientation="horizontal", **kwargs)
        
        self.headers = ["Pred", "N", "Num", "Color", "Parity", "H/L", "Dozen", "Column"]
        self.categories = ["number", "color", "parity", "high_low", "dozen", "column"]
        
        for i, header in enumerate(self.headers):
            w = 55 if i == 0 else 35
            lbl = ctk.CTkLabel(
                self,
                text=header,
                font=ctk.CTkFont(size=9, weight="bold"),
                width=w
            )
            lbl.grid(row=0, column=i, padx=1, pady=3, sticky="w")
        
        self.row_widgets: Dict[str, List[ctk.CTkLabel]] = {}
    
    def update_stats(self, stats: Dict[str, Dict], best_per_category: Dict[str, Dict]):
        for widgets in self.row_widgets.values():
            for w in widgets:
                w.destroy()
        self.row_widgets.clear()
        
        row = 1
        for predictor_name, predictor_stats in stats.items():
            widgets = []
            
            short_name = predictor_name.upper()[:6]
            name_lbl = ctk.CTkLabel(
                self,
                text=short_name,
                font=ctk.CTkFont(size=9),
                width=55,
                anchor="w"
            )
            name_lbl.grid(row=row, column=0, padx=1, pady=1, sticky="w")
            widgets.append(name_lbl)
            
            total_lbl = ctk.CTkLabel(
                self,
                text=str(predictor_stats.get("total", 0)),
                font=ctk.CTkFont(size=9),
                width=35
            )
            total_lbl.grid(row=row, column=1, padx=1, pady=1)
            widgets.append(total_lbl)
            
            for col_idx, cat in enumerate(self.categories):
                acc = predictor_stats.get(cat, 0)
                is_best = best_per_category.get(cat, {}).get("predictor") == predictor_name
                
                text_color = "#32CD32" if is_best else "white"
                font_weight = "bold" if is_best else "normal"
                
                acc_lbl = ctk.CTkLabel(
                    self,
                    text=f"{acc:.0f}",
                    font=ctk.CTkFont(size=9, weight=font_weight),
                    text_color=text_color,
                    width=35
                )
                acc_lbl.grid(row=row, column=col_idx + 2, padx=1, pady=1)
                widgets.append(acc_lbl)
            
            self.row_widgets[predictor_name] = widgets
            row += 1
    
    def clear(self):
        for widgets in self.row_widgets.values():
            for w in widgets:
                w.destroy()
        self.row_widgets.clear()


class IndividualPredictionsPanel(ctk.CTkFrame):
    
    RED_NUMBERS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.title_label = ctk.CTkLabel(
            self,
            text="Individual Predictions",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.title_label.pack(padx=10, pady=(10, 5))
        
        self.predictors_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.predictors_frame.pack(padx=10, pady=5, fill="x")
        
        self.predictor_widgets: Dict[str, Dict] = {}
        
        predictors = ["LSTM", "EXTRA_TREES", "BIAS"]
        for i, name in enumerate(predictors):
            self.predictors_frame.grid_columnconfigure(i, weight=1)
            
            frame = ctk.CTkFrame(self.predictors_frame, fg_color="gray25", corner_radius=8)
            frame.grid(row=0, column=i, padx=3, pady=3, sticky="nsew")
            
            name_lbl = ctk.CTkLabel(
                frame,
                text=name,
                font=ctk.CTkFont(size=10, weight="bold")
            )
            name_lbl.pack(pady=(5, 2))
            
            num_lbl = ctk.CTkLabel(
                frame,
                text="--",
                font=ctk.CTkFont(size=20, weight="bold"),
                text_color="gray"
            )
            num_lbl.pack()
            
            conf_lbl = ctk.CTkLabel(
                frame,
                text="--",
                font=ctk.CTkFont(size=9),
                text_color="gray"
            )
            conf_lbl.pack(pady=(0, 5))
            
            self.predictor_widgets[name.lower()] = {
                "frame": frame,
                "number": num_lbl,
                "confidence": conf_lbl
            }
    
    def _get_color_for_number(self, number: int) -> str:
        if number == 0:
            return "#228B22"
        elif number in self.RED_NUMBERS:
            return "#DC143C"
        else:
            return "#FFFFFF"
    
    def update_predictor(self, predictor_name: str, number: Optional[int], confidence: Optional[float], status: str = 'N/A'):
        key = predictor_name.lower()
        if key not in self.predictor_widgets:
            return
        
        widget = self.predictor_widgets[key]
        
        if number is not None and confidence is not None:
            color = self._get_color_for_number(number)
            widget["number"].configure(text=str(number), text_color=color)
            widget["confidence"].configure(text=f"{confidence:.1%}", text_color="white")
            widget["frame"].configure(fg_color="gray30")
        else:
            widget["number"].configure(text="--", text_color="gray")
            widget["confidence"].configure(text=status, text_color="gray")
            widget["frame"].configure(fg_color="gray20")
    
    def clear(self):
        for widget in self.predictor_widgets.values():
            widget["number"].configure(text="--", text_color="gray")
            widget["confidence"].configure(text="--", text_color="gray")
            widget["frame"].configure(fg_color="gray25")


class TrainingStatusBar(ctk.CTkFrame):
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.grid_columnconfigure(1, weight=1)
        
        self.status_label = ctk.CTkLabel(
            self,
            text="Ready",
            font=ctk.CTkFont(size=11)
        )
        self.status_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.progress_bar = ctk.CTkProgressBar(self, width=200)
        self.progress_bar.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.progress_bar.set(0)
        
        self.device_label = ctk.CTkLabel(
            self,
            text="Device: --",
            font=ctk.CTkFont(size=11)
        )
        self.device_label.grid(row=0, column=2, padx=10, pady=5, sticky="e")
    
    def set_status(self, status: str):
        self.status_label.configure(text=status)
    
    def set_progress(self, value: float):
        self.progress_bar.set(value)
    
    def set_device(self, device: str):
        self.device_label.configure(text=f"Device: {device}")
    
    def set_training(self, is_training: bool):
        if is_training:
            self.progress_bar.configure(mode="indeterminate")
            self.progress_bar.start()
        else:
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")
            self.progress_bar.set(0)
