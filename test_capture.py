#!/usr/bin/env python
"""
Test script for screen capture and OCR.
Run this to configure and test the capture region.
"""

import sys
import time
sys.path.insert(0, '.')

from src.capture import RouletteMonitor
from src.database.repository import RouletteRepository


class CaptureSession:
    def __init__(self):
        self.repo = RouletteRepository()
        self.session_id = None
        self.session_name = None
        self.numbers_added = 0
    
    def select_or_create_session(self) -> bool:
        print("\n" + "=" * 50)
        print("SESSION MANAGEMENT")
        print("=" * 50)
        
        sessions = self.repo.get_sessions()
        
        print("\nOptions:")
        print("  1. Create new session")
        print("  2. Delete a session")
        
        if sessions:
            print("\nExisting sessions:")
            for i, s in enumerate(sessions, 3):
                print(f"  {i}. {s['name']} ({s['spin_count']} spins) - {s['source']}")
        
        print("  0. Cancel")
        
        choice = input("\nSelect: ").strip()
        
        if choice == "0":
            return False
        
        if choice == "1":
            return self._create_new_session()
        
        if choice == "2":
            if sessions:
                self._delete_session(sessions)
                return self.select_or_create_session()
            else:
                print("No sessions to delete.")
                return self.select_or_create_session()
        
        try:
            idx = int(choice)
            if sessions and 3 <= idx < 3 + len(sessions):
                session = sessions[idx - 3]
                self.session_id = session['id']
                self.session_name = session['name']
                print(f"\nSelected: {self.session_name}")
                return True
        except ValueError:
            pass
        
        print("Invalid choice")
        return self.select_or_create_session()
    
    def _delete_session(self, sessions):
        print("\n--- Delete Session ---")
        print("Select session to delete:")
        for i, s in enumerate(sessions, 1):
            print(f"  {i}. {s['name']} ({s['spin_count']} spins)")
        print("  0. Cancel")
        
        choice = input("\nDelete session: ").strip()
        
        try:
            idx = int(choice)
            if idx == 0:
                return
            if 1 <= idx <= len(sessions):
                session = sessions[idx - 1]
                confirm = input(f"Delete '{session['name']}' with {session['spin_count']} spins? (yes/no): ").strip().lower()
                if confirm == "yes":
                    self.repo.db.delete_session(session['id'])
                    print(f"Session '{session['name']}' deleted.")
                else:
                    print("Cancelled.")
        except ValueError:
            print("Invalid choice")
    
    def _create_new_session(self) -> bool:
        print("\n--- Create New Session ---")
        name = input("Session name (Enter for auto): ").strip()
        casino = input("Casino name (optional): ").strip()
        
        if not name:
            from datetime import datetime
            name = f"Capture {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        session = self.repo.db.create_session(
            name=name,
            source="screen_capture",
            casino=casino,
            notes="Auto-captured via OCR"
        )
        self.session_id = session.id
        self.session_name = name
        print(f"\nCreated session: {name}")
        return True
    
    def add_number(self, number: int, *, event_id=None, observed_at=None):
        if self.session_id is None:
            print("[ERROR] No session selected!")
            return
        
        spin = self.repo.db.add_spin(self.session_id, number, event_id=event_id, timestamp=observed_at)
        self.numbers_added += 1
        return spin


def main():
    from src.console import configure_console
    configure_console()
    print("=" * 50)
    print("ROULETTE SCREEN CAPTURE")
    print("=" * 50)
    
    capture_session = CaptureSession()
    
    if not capture_session.select_or_create_session():
        print("No session selected. Exiting.")
        return
    
    def on_number(number: int, **metadata):
        spin = capture_session.add_number(number, **metadata)
        print(f">>> NUMBER {number} added to '{capture_session.session_name}' (total: {capture_session.numbers_added})")
        return spin

    def on_observation(observation):
        capture_session.repo.db.record_capture_observation(
            capture_session.session_id, observation.numbers, observation.confidence,
            event_id=observation.event_id, observed_at=observation.observed_at,
            status=observation.status, reason=observation.reason)
    
    monitor = RouletteMonitor(on_number_detected=on_number, on_observation=on_observation)
    
    existing_numbers = capture_session.repo.get_numbers_by_session(capture_session.session_id)
    monitor.set_session_numbers(existing_numbers)
    print(f"Loaded {len(existing_numbers)} existing numbers for reconciliation")
    
    while True:
        reconnect_status = "ON" if monitor.has_reconnect_region() and monitor.auto_reconnect_enabled else "OFF"
        history_status = "ON" if monitor.has_history_region() else "OFF"
        print(f"\n[Session: {capture_session.session_name} | Numbers: {capture_session.numbers_added} | Reconnect: {reconnect_status} | History: {history_status}]")
        print("\nOptions:")
        print("  1. Select capture region (roulette number)")
        print("  2. Test capture (read number once)")
        print("  3. Start monitoring")
        print("  4. Show current config")
        print("  5. Save capture preview")
        print("  6. Change session")
        print("  7. Setup auto-reconnect (select button region)")
        print("  8. Setup history region (for reconciliation)")
        print("  9. Set inactivity timeout")
        print("  0. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == "1":
            print("\nA fullscreen window will appear.")
            print("Click and drag to select the region with the roulette numbers.")
            print("Press ESC to cancel.")
            input("Press Enter to continue...")
            
            success = monitor.select_region()
            if success:
                print("Region saved!")
            else:
                print("Selection cancelled.")
        
        elif choice == "2":
            print("\nTesting capture...")
            number = monitor.test_capture()
            if number is not None:
                print(f"Detected: {number}")
            else:
                print("No number detected. Try adjusting the region.")
        
        elif choice == "3":
            print("\nStarting monitoring...")
            print("Press Ctrl+C to stop.")
            try:
                monitor.start()
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                monitor.stop()
                print(f"\nMonitoring stopped. Total numbers added this session: {capture_session.numbers_added}")
                print(f"Reconnects: {monitor.reconnect_count}")
        
        elif choice == "4":
            stats = monitor.get_stats()
            print(f"\nCurrent configuration:")
            print(f"  Session: {capture_session.session_name}")
            print(f"  Capture region: ({stats['region']['x']}, {stats['region']['y']}) {stats['region']['width']}x{stats['region']['height']}")
            print(f"  Detections: {stats['detection_count']}")
            print(f"  Reconnects: {stats['reconnect_count']}")
            print(f"  Reconciled: {stats['reconciled_count']}")
            print(f"  Last detection: {stats['last_detection']}")
            print(f"  Inactivity timeout: {monitor.config.inactivity_timeout}s")
            if monitor.has_reconnect_region():
                print(f"  Reconnect region: ({monitor.config.reconnect_x}, {monitor.config.reconnect_y}) {monitor.config.reconnect_width}x{monitor.config.reconnect_height}")
            else:
                print("  Reconnect region: Not configured")
            if monitor.has_history_region():
                print(f"  History region: ({monitor.config.history_x}, {monitor.config.history_y}) {monitor.config.history_width}x{monitor.config.history_height}")
                print(f"  Reconcile after reconnect: {monitor.config.reconcile_after_reconnect}")
                print(f"  Reconcile count (max numbers): {monitor.config.reconcile_count}")
            else:
                print("  History region: Not configured")
        
        elif choice == "5":
            print("\nCapturing region...")
            img = monitor.screen_capture.capture()
            if img:
                preview_path = "data/capture_preview.png"
                img.save(preview_path)
                print(f"Preview saved to: {preview_path}")
                print("Open this file to see exactly what's being captured.")
            else:
                print("Failed to capture. Make sure region is configured (option 1).")
        
        elif choice == "6":
            monitor.stop()
            if capture_session.select_or_create_session():
                monitor = RouletteMonitor(on_number_detected=on_number, on_observation=on_observation)
                existing_numbers = capture_session.repo.get_numbers_by_session(capture_session.session_id)
                monitor.set_session_numbers(existing_numbers)
                print(f"Loaded {len(existing_numbers)} existing numbers for reconciliation")
        
        elif choice == "7":
            print("\nSetup auto-reconnect region.")
            print("Select the area where the 'Reconectar' button appears.")
            print("When no new numbers are detected for the timeout period,")
            print("the system will click the center of this region.")
            input("Press Enter to continue...")
            
            success = monitor.select_reconnect_region()
            if success:
                print("Auto-reconnect configured!")
            else:
                print("Selection cancelled.")
        
        elif choice == "8":
            print("\nSetup history region for reconciliation.")
            print("Select the area where the last numbers appear.")
            print("(Newest number on the LEFT, oldest on the RIGHT)")
            print("After a reconnect, the system will read this area")
            print("to recover any numbers missed during the popup.")
            input("Press Enter to continue...")
            
            success = monitor.select_history_region()
            if success:
                print("History region configured!")
            else:
                print("Selection cancelled.")
        
        elif choice == "9":
            current = monitor.config.inactivity_timeout
            print(f"\nCurrent inactivity timeout: {current}s")
            new_timeout = input("Enter new timeout in seconds (default 60): ").strip()
            try:
                timeout = float(new_timeout) if new_timeout else 60.0
                monitor.set_inactivity_timeout(timeout)
            except ValueError:
                print("Invalid value.")
        
        elif choice == "0":
            print(f"Session '{capture_session.session_name}' - Total numbers added: {capture_session.numbers_added}")
            print("Bye!")
            break
        
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
