import customtkinter as ctk
from typing import Optional, List
from pathlib import Path
import threading
from queue import SimpleQueue, Empty
from src.datasets import EvaluationSession
from src.utils.evaluation_harness import EvaluationConfig, evaluate_walk_forward, format_evaluation_report

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
    
    def __init__(self, model_path: Optional[str] = None, db_path: Optional[str] = None, device: str = 'auto'):
        super().__init__()
        
        self.title("RL Roulette - Prediction System")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        
        if model_path is None:
            default_path = Path(__file__).parent.parent.parent / self.DEFAULT_MODEL
            if default_path.exists():
                model_path = str(default_path)
        
        self.db = Database(db_path)
        self._events = SimpleQueue()
        self._jobs = {}
        self._job_counter = 0
        self._closing = False
        self.engine = PredictionEngine(model_path=model_path, device=device)
        self.current_session_id: Optional[int] = None
        
        self._setup_grid()
        self._create_widgets()
        self._load_initial_data()
        
        self.protocol('WM_DELETE_WINDOW', self._close)
        self._poll_after = self.after(50, self._poll_jobs)
    
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

        self.heatmaps_btn = ctk.CTkButton(
            self.buttons_frame,
            text="Heatmaps",
            command=self._on_heatmaps_clicked
        )
        self.heatmaps_btn.grid(row=1, column=0, padx=0, pady=(5, 0), sticky="ew")
        self.cancel_btn = ctk.CTkButton(self.buttons_frame, text='Cancel', command=self._cancel_jobs)
        self.cancel_btn.grid(row=1, column=1, padx=5, pady=(5, 0), sticky='ew')
        
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
        self.policy_label = ctk.CTkLabel(self.right_panel, text='DQN policy: no checkpoint selected', anchor='w')
        self.policy_label.grid(row=3, column=0, padx=10, pady=5, sticky='ew')
        
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
        self._cancel_jobs()
        self.engine.load_history(numbers, session_id=session_id)
        self.stats_table.clear()
        
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
    
    def _on_new_session(self, emit_initial=True):
        dialog = ctk.CTkInputDialog(
            text="Enter session name:",
            title="New Session"
        )
        name = dialog.get_input()
        
        if name:
            session = self.db.create_session(name, source="gui")
            self.current_session_id = session.id
            
            self._cancel_jobs()
            self.engine.load_history([], session_id=session.id)
            
            sessions = self._get_sessions()
            self.session_panel.update_sessions(sessions)
            self.session_panel.session_var.set(name)
            self.session_panel.set_spin_count(0)
            
            self.prediction_display.clear()
            self.probability_bars.clear()
            self.top_numbers_fan.clear()
            self.stats_table.clear()
            if emit_initial:
                self._update_predictions()
    
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
            self._cancel_jobs()
            self.engine.load_history([], session_id=None)
            
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
            self._on_new_session(emit_initial=False)
            if self.current_session_id is None:
                return
        try:
            self.db.add_spin(self.current_session_id, number)
        except Exception as error:
            self.status_bar.set_status(f'Number was not saved: {error}')
            return
        self.engine.add_number(number)
        self.session_panel.set_spin_count(len(self.engine.history))
        self.number_input.update_last_numbers(self.engine.history)
        self._update_predictions()
        self._update_stats_display()

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
                    None,
                    status=self.engine.statuses[pred_type].state.value
                )
        
        consensus = self.engine.get_consensus_prediction(predictions)
        if self.current_session_id is not None:
            emitted = dict(predictions)
            emitted[consensus.predictor] = consensus
            try:
                self.db.emit_predictions(self.current_session_id, emitted, expected_count=len(self.engine.history))
            except Exception as error:
                self.status_bar.set_status(f'Predictions were not saved: {error}')
                self.prediction_display.clear()
                return
        decision = self.engine.get_policy_decision()
        if decision.action is None:
            self.policy_label.configure(text=f'DQN policy: {decision.status.state.value} — {decision.status.reason}')
        else:
            action = 'PASS' if decision.action == 46 else str(decision.action)
            self.policy_label.configure(text=f'DQN policy: action {action}, stake {decision.stake:g}')
        
        if consensus is None:
            self.prediction_display.clear()
            self.probability_bars.clear()
            self.top_numbers_fan.clear()
            return
        
        self.prediction_display.update_prediction(
            number=consensus.number.value,
            confidence=consensus.number.probability,
            predictor="Fair baseline" if consensus.predictor == PredictorType.FAIR else "Consensus",
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
        self.stats_table.clear()
        if self.current_session_id is None:
            return
        stats = self.db.get_predictor_stats(self.current_session_id)
        summary = {item['predictor_type']: item['accuracies'] | {'total': item['total']} for item in stats}
        self.stats_table.update_stats(summary, self.db.get_best_predictor_per_category(self.current_session_id))

    def _register_job(self, kind):
        if kind in self._jobs:
            return None
        self._job_counter += 1
        token = (self.current_session_id, self.engine.data_version, self._job_counter)
        cancel = threading.Event()
        self._jobs[kind] = (token, cancel)
        return token, cancel

    def _run_job(self, kind, work):
        registered = self._register_job(kind)
        if registered is None:
            return
        token, cancel = registered
        def worker():
            result, error = None, None
            try:
                result = work(cancel)
            except Exception as caught:
                error = str(caught)
            finally:
                self._events.put((kind, token, result, error))
        threading.Thread(target=worker, daemon=True).start()

    def _poll_jobs(self):
        if self._closing:
            return
        try:
            while True:
                kind, token, result, error = self._events.get_nowait()
                self._finish_job(kind, token, result, error)
        except Empty:
            pass
        self._poll_after = self.after(50, self._poll_jobs)

    def _finish_job(self, kind, token, result, error):
        current = self._jobs.get(kind)
        if current is None or current[0] != token:
            return
        self._jobs.pop(kind)
        buttons = {'train': (self.train_btn, 'Train Models'), 'backtest': (self.backtest_btn, 'Backtest'),
                   'heatmaps': (self.heatmaps_btn, 'Heatmaps')}
        button, text = buttons[kind]
        button.configure(state='normal', text=text)
        if kind == 'train':
            self.status_bar.set_training(False)
        if token[:2] != (self.current_session_id, self.engine.data_version):
            return
        if current[1].is_set():
            self.status_bar.set_status('Cancelled')
            return
        if error:
            self.status_bar.set_status(f'{kind.capitalize()} failed: {error}')
            return
        if kind == 'train':
            self._training_finished_callback()
        elif kind == 'backtest':
            self._backtest_finished(result)
        else:
            self._heatmaps_finished(result, None)

    def _cancel_jobs(self):
        self.engine.cancel_training()
        for _, cancel in self._jobs.values():
            cancel.set()

    def _close(self):
        self._closing = True
        self._cancel_jobs()
        self.after_cancel(self._poll_after)
        self.destroy()

    def _on_train_clicked(self):
        if self.engine.training_in_progress or any(kind in self._jobs for kind in ('train', 'backtest')):
            return
        token, cancel = self._register_job('train')
        self.train_btn.configure(state='disabled', text='Training...')
        self.status_bar.set_training(True)
        self.status_bar.set_status('Training models...')
        try:
            self.engine.train_async(epochs=30, on_complete=lambda: self._events.put(
                ('train', token, self.engine.last_training_result, None)))
        except Exception as error:
            self._finish_job('train', token, None, str(error))

    def _on_backtest_clicked(self):
        if self.current_session_id is None or any(kind in self._jobs for kind in ('train', 'backtest')):
            return
        session = self.db.get_session(self.current_session_id)
        if session is None:
            return
        snapshot = EvaluationSession(str(session.id), tuple(spin.number for spin in session.spins),
                                     tuple(str(spin.id) for spin in session.spins), session.source)
        self.backtest_btn.configure(state='disabled', text='Testing...')
        self.status_bar.set_status('Running temporal evaluation...')
        config = EvaluationConfig(model_path=self.engine.model_path, device=self.engine.device,
                                  models=('lstm', 'extra_trees', 'bias') + (('dqn',) if self.engine.model_path else ()))
        self._run_job('backtest', lambda cancel: evaluate_walk_forward([snapshot], config, cancel_event=cancel))

    def _on_heatmaps_clicked(self):
        if self.current_session_id is None:
            self.status_bar.set_status("Select or create a session before generating heatmaps")
            return

        self.heatmaps_btn.configure(state="disabled", text="Generating heatmaps...")
        self.status_bar.set_status("Generating heatmap report...")

        if 'heatmaps' in self._jobs:
            return
        session = self.db.get_session(self.current_session_id)
        if session is None:
            self.heatmaps_btn.configure(state='normal', text='Heatmaps')
            return
        numbers = [spin.number for spin in session.spins]
        session_id = session.id
        def work(cancel):
            from src.utils.heatmaps import HeatmapConfig, generate_heatmap_report
            return generate_heatmap_report(numbers, HeatmapConfig(
                output_dir=str(Path('reports') / 'heatmaps' / f'session_{session_id}'), show=False),
                source_label=f'session {session_id}')
        self._run_job('heatmaps', work)

    def _heatmaps_finished(self, result, error):
        self.heatmaps_btn.configure(state="normal", text="Heatmaps")
        if error is not None:
            self.status_bar.set_status(f"Heatmaps failed: {error}")
            return
        self.status_bar.set_status(
            f"Heatmaps ready: {len(result.output_paths)} files, {len(result.anomalies)} alerts"
        )
        self._show_heatmap_report(result)

    def _show_heatmap_report(self, result):
        window = ctk.CTkToplevel(self)
        window.title("Heatmap Report")
        window.geometry("720x520")
        window.grid_columnconfigure(0, weight=1)
        window.grid_rowconfigure(1, weight=1)

        title = ctk.CTkLabel(
            window,
            text=f"Heatmaps - {result.source_label}",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="w")

        textbox = ctk.CTkTextbox(window, wrap="word")
        textbox.grid(row=1, column=0, padx=16, pady=(0, 16), sticky="nsew")

        lines = [
            f"Total spins: {result.total_spins}",
            "",
            "Generated files:",
        ]
        for name, path in result.output_paths.items():
            lines.append(f"- {name}: {path}")
        if result.anomalies:
            lines.extend(["", "Top anomalies:"])
            for anomaly in result.anomalies[:12]:
                lines.append(
                    f"- {anomaly.severity} | {anomaly.window_label} | "
                    f"{anomaly.feature} | z={anomaly.z_score:+.2f} | "
                    f"obs={anomaly.observed_rate:.3f} exp={anomaly.expected_rate:.3f}"
                )
        else:
            lines.extend(["", "No anomalies detected."])
        if result.warnings:
            lines.extend(["", "Warnings:"])
            lines.extend(f"- {warning}" for warning in result.warnings)

        textbox.insert("1.0", "\n".join(lines))
        textbox.configure(state="disabled")

    def _backtest_finished(self, result):
        self.status_bar.set_status(f'Evaluation finished: {result.folds} folds')
        window = ctk.CTkToplevel(self)
        window.title('Temporal evaluation')
        window.geometry('1000x650')
        textbox = ctk.CTkTextbox(window, wrap='none')
        textbox.pack(fill='both', expand=True, padx=12, pady=12)
        textbox.insert('1.0', format_evaluation_report(result))
        textbox.configure(state='disabled')

    def _training_finished_callback(self):
        status = self.engine.get_status()
        details = [f"{name}: {status['models'][name]['state']} {status['models'][name]['reason']}"
                   for name in ('lstm', 'extra_trees')]
        self.status_bar.set_status(' | '.join(details))
        self._update_predictions()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', nargs='?')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--database')
    args = parser.parse_args()
    app = RouletteGUI(model_path=args.model_path, db_path=args.database, device=args.device)
    app.mainloop()


if __name__ == "__main__":
    main()
