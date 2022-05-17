from app import db

db.create_all()
if __name__ == '__main__':
    try:
        db.create_all()
        print('Database created.')
    except Exception as e:
        print('Database not created.')
        print(e)
