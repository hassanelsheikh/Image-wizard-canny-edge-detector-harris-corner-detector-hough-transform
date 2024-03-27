from PyQt5 import QtWidgets
from view.main_window import Ui_MainWindow
from controller.controller import Controller
import sys

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # Create an instance of the main window UI
    main_window = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(main_window)
    
    # Create an instance of the Controller and pass the UI to it
    controller = Controller(ui)
    
    # Show the main window
    main_window.show()
    
    sys.exit(app.exec_())
