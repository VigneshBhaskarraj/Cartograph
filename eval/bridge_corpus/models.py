"""A tiny SQLAlchemy-style app layer mapped onto schema.sql, for the bridge eval."""


class Base:
    pass


class User(Base):
    __tablename__ = "users"

    def display_name(self) -> str:
        # WHY: show email until profile names exist.
        return self.email


class Order(Base):
    __tablename__ = "orders"

    def is_paid(self) -> bool:
        return self.total is not None


def orders_for_user(session, user_id: int):
    """Fetch a user's orders."""
    return session.query(Order).filter(Order.user_id == user_id).all()
