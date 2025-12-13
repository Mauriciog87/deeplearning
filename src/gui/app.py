import customtkinter as ctk
from typing import Optional, List
from pathlib import Path
import threading

from src.gui.components import (
    SessionPanel,
    NumberInputPanel,
    PredictionDisplay,
    ProbabilityBars,
    TopNumbersFan,
    PredictorStatsTable,
    TrainingStatusBar,
    IndividualPredictionsPanel
)
from src.engine.prediction_engine import PredictionEngine, PredictorType
from src.database.models import Database


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class RouletteGUI(ctk.CTk):
    
    DEFAULT_MODEL = "models/roulette_agent.pt"
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__()
        
        self.title("RL Roulette - Prediction System")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        
        if model_path is None:
            default_path = Path(__file__).parent.parent.parent / self.DEFAULT_MODEL
            if default_path.exists():
                model_path = str(default_path)
        
        self.db = Database()
        self.engine = PredictionEngine(model_path=model_path)
        self.current_session_id: Optional[int] = None
        
        self._setup_grid()
        self._create_widgets()
        self._load_initial_data()
        
        self.engine.on_training_complete = self._on_training_complete
    
    def _setup_grid(self):
        self.grid_columnconfigure(0, weight=0, minsize=300)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
    
    def _create_widgets(self):
        self.left_panel = ctk.CTkFrame(self)
        self.left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.left_panel.grid_rowconfigure(4, weight=1)
        self.left_panel.grid_columnconfigure(0, weight=1)
        
        sessions = self._get_sessions()
        self.session_panel = SessionPanel(
            self.left_panel,
            sessions=sessions,
            on_session_change=self._on_session_change,
            on_new_session=self._on_new_session,
            on_load_session=self._on_load_session,
            on_delete_session=self._on_delete_session
        )
        self.session_panel.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.number_input = NumberInputPanel(
            self.left_panel,
            on_add_number=self._on_add_number
        )
        self.number_input.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        self.prediction_display = PredictionDisplay(self.left_panel)
        self.prediction_display.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        
        self.buttons_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.buttons_frame.grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        self.buttons_frame.grid_columnconfigure((0, 1), weight=1)
        
        self.train_btn = ctk.CTkButton(
            self.buttons_frame,
            text="Train Models",
            command=self._on_train_clicked
        )
        self.train_btn.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="ew")
        
        self.backtest_btn = ctk.CTkButton(
            self.buttons_frame,
            text="Backtest",
            command=self._on_backtest_clicked
        )
        self.backtest_btn.grid(row=0, column=1, padx=(5, 0), pady=5, sticky="ew")
        
        self.stats_table = PredictorStatsTable(
            self.left_panel,
            label_text="Predictor Accuracy"
        )
        self.stats_table.grid(row=4, column=0, padx=5, pady=5, sticky="nsew")
        
        self.right_panel = ctk.CTkFrame(self)
        self.right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.right_panel.grid_rowconfigure(2, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)
        
        self.individual_predictions = IndividualPredictionsPanel(self.right_panel)
        self.individual_predictions.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.top_numbers_fan = TopNumbersFan(self.right_panel, count=10)
        self.top_numbers_fan.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        self.probability_bars = ProbabilityBars(self.right_panel)
        self.probability_bars.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")
        
        self.status_bar = TrainingStatusBar(self)
        self.status_bar.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")
        
        status = self.engine.get_status()
        self.status_bar.set_device(status.get("device", "Unknown"))
    
    def _get_sessions(self) -> List[dict]:
        sessions = self.db.get_all_sessions()
        return [{"id": s["id"], "name": s["name"]} for s in sessions]
    
    def _load_initial_data(self):
        sessions = self._get_sessions()
        if sessions:
            first_session = sessions[0]
            self.current_session_id = first_session["id"]
            self._load_session_data(self.current_session_id)
    
    def _load_session_data(self, session_id: int):
        session = self.db.get_session(session_id)
        if not session:
            return
        
        numbers = [spin.number for spin in session.spins]
        self.engine.load_history(numbers)
        
        self.session_panel.set_spin_count(len(numbers))
        self.number_input.update_last_numbers(numbers)
        
        stats = self.db.get_predictor_stats(session_id)
        if stats:
            stats_dict = {s["predictor_type"]: s["accuracies"] | {"total": s["total"]} for s in stats}
            best = self.db.get_best_predictor_per_category(session_id)
            self.stats_table.update_stats(stats_dict, best)
        
        self._update_predictions()
    
    def _on_session_change(self, session_name: str):
        sessions = self._get_sessions()
        for s in sessions:
            if s["name"] == session_name:
                self.current_session_id = s["id"]
                self._load_session_data(self.current_session_id)
                break
    
    def _on_new_session(self):
        dialog = ctk.CTkInputDialog(
            text="Enter session name:",
            title="New Session"
        )
        name = dialog.get_input()
        
        if name:
            session = self.db.create_session(name, source="gui")
            self.current_session_id = session.id
            
            self.engine.load_history([])
            
            sessions = self._get_sessions()
            self.session_panel.update_sessions(sessions)
            self.session_panel.session_var.set(name)
            self.session_panel.set_spin_count(0)
            
            self.prediction_display.clear()
            self.probability_bars.clear()
            self.top_numbers_fan.clear()
            self.stats_table.clear()
    
    def _on_load_session(self):
        sessions = self._get_sessions()
        if not sessions:
            return
        
        self._on_session_change(self.session_panel.session_var.get())
    
    def _on_delete_session(self):
        if self.current_session_id is None:
            return
        
        session_name = self.session_panel.session_var.get()
        
        dialog = ctk.CTkInputDialog(
            text=f"Type 'DELETE' to confirm deletion of:\n'{session_name}'",
            title="Delete Session"
        )
        result = dialog.get_input()
        
        if result and result.upper() == "DELETE":
            self.db.delete_session(self.current_session_id)
            
            self.current_session_id = None
            self.engine.history.clear()
            
            sessions = self._get_sessions()
            self.session_panel.update_sessions(sessions)
            
            if sessions:
                self._on_session_change(sessions[0]['name'])
            else:
                self.session_panel.set_spin_count(0)
                self.prediction_display.clear()
                self.probability_bars.clear()
                self.top_numbers_fan.clear()
                self.stats_table.clear()
    
    def _on_add_number(self, number: int):
        if self.current_session_id is None:
            self._on_new_session()
            if self.current_session_id is None:
                return
        
        self.db.add_spin(self.current_session_id, number)
        
        predictions = self.engine.predict_all()
        
        self.engine.validate_prediction(number)
        
        self.engine.add_number(number)
        
        self._save_predictor_stats()
        
        self.session_panel.set_spin_count(len(self.engine.history))
        self.number_input.update_last_numbers(self.engine.history)
        
        self._update_predictions()
        self._update_stats_display()
    
    def _save_predictor_stats(self):
        if self.current_session_id is None:
            return
        
        for pred_type, stats in self.engine.stats.items():
            d1 = stats.dozen1_correct
            d2 = stats.dozen2_correct
            d3 = stats.dozen3_correct
            c1 = stats.column1_correct
            c2 = stats.column2_correct
            c3 = stats.column3_correct
            
            self.db.upsert_predictor_stats(
                session_id=self.current_session_id,
                predictor_type=pred_type.value,
                total_predictions=stats.total_predictions,
                number_correct=stats.number_correct,
                color_correct=stats.color_correct,
                parity_correct=stats.parity_correct,
                high_low_correct=stats.high_low_correct,
                dozen_correct=d1 + d2 + d3,
                column_correct=c1 + c2 + c3
            )
    
    def _update_predictions(self):
        predictions = self.engine.predict_all()
        
        for pred_type, pred in predictions.items():
            if pred is not None:
                self.individual_predictions.update_predictor(
                    pred_type.value,
                    pred.number.value,
                    pred.number.probability
                )
            else:
                self.individual_predictions.update_predictor(
                    pred_type.value,
                    None,
                    None
                )
        
        consensus = self.engine.get_consensus_prediction()
        
        if consensus is None:
            self.prediction_display.clear()
            self.probability_bars.clear()
            self.top_numbers_fan.clear()
            return
        
        self.prediction_display.update_prediction(
            number=consensus.number.value,
            confidence=consensus.number.probability,
            predictor="Consensus",
            top_numbers=consensus.top_numbers[:3]
        )
        
        self.probability_bars.update_category(
            "color",
            consensus.color.value,
            consensus.color.probability,
            consensus.color.all_probabilities
        )
        self.probability_bars.update_category(
            "parity",
            consensus.parity.value,
            consensus.parity.probability,
            consensus.parity.all_probabilities
        )
        self.probability_bars.update_category(
            "high_low",
            consensus.high_low.value,
            consensus.high_low.probability,
            consensus.high_low.all_probabilities
        )
        self.probability_bars.update_category(
            "dozen",
            consensus.dozen.value,
            consensus.dozen.probability,
            consensus.dozen.all_probabilities
        )
        self.probability_bars.update_category(
            "column",
            consensus.column.value,
            consensus.column.probability,
            consensus.column.all_probabilities
        )
        
        self.top_numbers_fan.update_numbers(consensus.top_numbers)
    
    def _update_stats_display(self):
        stats_summary = self.engine.get_stats_summary()
        
        best = {}
        categories = ["number", "color", "parity", "high_low", "dozen", "column"]
        for cat in categories:
            best_pred = self.engine.get_best_predictor(cat)
            if best_pred:
                best[cat] = {"predictor": best_pred.value}
            else:
                best[cat] = {"predictor": None}
        
        self.stats_table.update_stats(stats_summary, best)
    
    def _on_train_clicked(self):
        if self.engine.training_in_progress:
            return
        
        self.train_btn.configure(state="disabled", text="Training...")
        self.status_bar.set_status("Training models...")
        self.status_bar.set_training(True)
        
        self.engine.train_async(epochs=50, on_complete=self._on_training_complete)
    
    def _on_backtest_clicked(self):
        if self.engine.training_in_progress:
            return
        
        self.backtest_btn.configure(state="disabled", text="Testing...")
        self.status_bar.set_status("Running backtest...")
        
        def backtest_worker():
            result = self.engine.backtest_history(min_history=20)
            self.after(0, lambda: self._backtest_finished(result))
        
        import threading
        thread = threading.Thread(target=backtest_worker, daemon=True)
        thread.start()
    
    def _backtest_finished(self, result: dict):
        self.backtest_btn.configure(state="normal", text="Backtest")
        
        if "models_status" in result:
            status = result["models_status"]
            lstm = "✓" if status["lstm_trained"] else "✗"
            et = "✓" if status["extra_trees_trained"] else "✗"
            dqn = "✓" if status["dqn_loaded"] else "✗"
            self.status_bar.set_status(f"Backtest done | LSTM:{lstm} ET:{et} DQN:{dqn}")
        else:
            self.status_bar.set_status("Backtest complete")
        
        self._update_stats_display()
        self._save_predictor_stats()
    
    def _on_training_complete(self):
        self.after(0, self._training_finished_callback)
    
    def _training_finished_callback(self):
        self.train_btn.configure(state="normal", text="Train Models")
        self.status_bar.set_status("Training complete")
        self.status_bar.set_training(False)
        
        status = self.engine.get_status()
        lstm_status = "✓" if status["lstm_trained"] else "✗"
        et_status = "✓" if status["extra_trees_trained"] else "✗"
        self.status_bar.set_status(f"LSTM: {lstm_status} | ExtraTrees: {et_status}")
        
        self._update_predictions()


def main():
    import sys
    
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    app = RouletteGUI(model_path=model_path)
    app.mainloop()


if __name__ == "__main__":
    main()
