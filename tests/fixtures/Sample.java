package com.acme.shop;

import com.acme.shop.Auditable;
import javax.persistence.Entity;
import javax.persistence.Table;
import javax.persistence.Column;

@Entity
@Table(name = "orders")
public class Order extends BaseEntity implements Auditable {
    @Column(name = "total_amount")
    private double total;

    public boolean isPaid() {
        return this.checkTotal();
    }

    private boolean checkTotal() {
        return total > 0;
    }

    public static class Builder {
        public Order build() {
            return new Order();
        }
    }
}

class BaseEntity {
    public void audit() {}
}

interface Auditable {
    void audit();
}
